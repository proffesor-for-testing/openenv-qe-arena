"""OpenEnv environment: virtual files, exact-replacement edits, sandboxed tests."""

from __future__ import annotations

import json
import secrets
from typing import Any, Literal
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

from .tasks import MODULE, TASK_IDS, Task, grade_bugfix, grade_mutant_kill, make_task, run_visible

MAX_STEPS = 12
MAX_TEST_RUNS = 3
MAX_EDIT_BYTES = 4096
MAX_FILE_BYTES = 6000
MAX_ANSWER_BYTES = 8192
MAX_FREE_REFUSALS = 4  # over-budget edits that do not use a step; later ones do, so episodes stay bounded


class QEAction(Action):
    op: Literal["read", "run_tests", "edit", "submit"] = Field(description=(
        "One action per reply. Start with read path * to see every file. "
        "read: show a file. run_tests: run the visible tests on your edited code (bug-fix tasks). "
        "edit: replace old with new in a file (bug-fix tasks). submit: end the episode and get graded."))
    path: str = Field(default="", description="read/edit: the file name, or * with read for all files.")
    old: str = Field(default="", description="edit: exact text copied from the file; it must occur exactly once.")
    new: str = Field(default="", description="edit: the text that replaces old.")
    answer: dict[str, Any] = Field(default_factory=dict, description=(
        "submit: {} grades your edited file in a bug-fix task; "
        '{"tests": [{"args": [...], "expected": ...}]} in a mutant-kill task.'))


class QEObservation(Observation):
    task_id: str = ""
    prompt: str = ""
    files: list[str] = Field(default_factory=list)
    content: str = ""
    error: str = ""
    remaining_steps: int = 0


class QEEnvironment(Environment[QEAction, QEObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, corpus_pool: tuple[str, ...] | None = None, fault_kind: str | None = None):
        """``corpus_pool`` and ``fault_kind`` are for calibration only; the server passes nothing."""
        super().__init__()
        self._pool = corpus_pool
        self._fault_kind = fault_kind
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: Task | None = None
        self._files: dict[str, str] = {}
        self._done = False
        self._reward = 0.0
        self._test_runs = 0
        self._edit_spent = 0
        self._refusals = 0

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> QEObservation:
        task_id = kwargs.get("task_id", TASK_IDS[0])
        if seed is None:
            seed = secrets.randbelow(2**63)
        kind = self._fault_kind if task_id.startswith("qe_bugfix") else None
        self._task = make_task(task_id, seed, pool=self._pool, fault_kind=kind)
        self._files = dict(self._task.files)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._done, self._reward, self._test_runs = False, 0.0, 0
        self._edit_spent, self._refusals = 0, 0
        return self._observe(prompt=self._task.prompt, files=sorted(self._files))

    def _observe(self, **fields: Any) -> QEObservation:
        return QEObservation(
            task_id=self._task.task_id if self._task else "", done=self._done, reward=self._reward,
            remaining_steps=max(0, MAX_STEPS - self._state.step_count), **fields)

    def step(self, action: QEAction, timeout_s: float | None = None, **kwargs: Any) -> QEObservation:
        if self._task is None:
            raise RuntimeError("reset required")
        if self._done:
            return self._observe(error="episode is terminal; reset before another attempt")
        if self._free_refusal(action):
            return self._refused(len(action.old) + len(action.new), self._task.private["edit_budget"])
        self._state.step_count += 1
        family = self._task.family
        if action.op == "submit":
            return self._submit(action)
        if self._state.step_count >= MAX_STEPS:
            self._done = True
            return self._observe(error="step budget exhausted; reward 0")
        if action.op == "read":
            if action.path == "*":
                return self._observe(content="\n".join(f"=== {n} ===\n{t}" for n, t in sorted(self._files.items())))
            if action.path not in self._files:
                return self._observe(error=f"unknown file; files are {sorted(self._files)}")
            return self._observe(content=self._files[action.path])
        if family != "qe_bugfix":
            return self._observe(error=f"{action.op} is not available in this task; read the files, then submit")
        if action.op == "run_tests":
            if self._test_runs >= MAX_TEST_RUNS:
                return self._observe(error="no test runs left; submit when ready")
            self._test_runs += 1
            return self._observe(content=run_visible(self._task, self._files[MODULE]))
        return self._edit(action)

    def _free_refusal(self, action: QEAction) -> bool:
        """An over-budget edit is refused before it can use a step, up to MAX_FREE_REFUSALS times."""
        budget = self._task.private.get("edit_budget")
        if budget is None or action.op != "edit" or action.path != MODULE or not action.old:
            return False
        if self._edit_spent + len(action.old) + len(action.new) <= budget:
            return False
        self._refusals += 1
        return self._refusals <= MAX_FREE_REFUSALS

    def _refused(self, cost: int, budget: int) -> QEObservation:
        return self._observe(error=f"edit refused: it costs {cost} characters (len(old) + len(new)) and "
                                   f"{budget - self._edit_spent} of the {budget} budget are left; nothing was changed")

    def _edit(self, action: QEAction) -> QEObservation:
        if action.path != MODULE:
            return self._observe(error=f"only {MODULE} can be edited")
        if not action.old or len(action.old) + len(action.new) > MAX_EDIT_BYTES:
            return self._observe(error="old must be non-empty and the edit small")
        budget = self._task.private.get("edit_budget")
        cost = len(action.old) + len(action.new)
        if budget is not None and self._edit_spent + cost > budget:
            return self._refused(cost, budget)
        text = self._files[MODULE]
        count = text.count(action.old)
        if count != 1:
            return self._observe(error=f"old occurs {count} times; it must occur exactly once")
        updated = text.replace(action.old, action.new, 1)
        try:
            size = len(updated.encode("utf-8"))
        except UnicodeEncodeError:
            return self._observe(error="edit text must be valid UTF-8")
        if size > MAX_FILE_BYTES:
            return self._observe(error="file too large after edit")
        self._files[MODULE] = updated
        message = f"edited {MODULE}; {len(updated.splitlines())} lines"
        if budget is not None:
            self._edit_spent += cost
            message += f"; edit budget left: {budget - self._edit_spent} of {budget} characters"
        return self._observe(content=message)

    def _submit(self, action: QEAction) -> QEObservation:
        self._done = True
        try:
            if self._task.family == "qe_bugfix":
                self._reward = grade_bugfix(self._task, self._files[MODULE])
            else:
                encoded = json.dumps(action.answer, allow_nan=False, separators=(",", ":"))
                too_big = len(encoded.encode()) > MAX_ANSWER_BYTES
                self._reward = 0.0 if too_big else grade_mutant_kill(self._task, action.answer)
        except Exception:  # noqa: BLE001 - any grading failure on a model answer scores 0, never an error frame
            self._reward = 0.0
        return self._observe()

    @property
    def state(self) -> State:
        return self._state.model_copy(deep=True)
