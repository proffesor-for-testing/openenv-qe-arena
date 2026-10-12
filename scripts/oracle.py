"""Reference solutions built from private task data, for verifier controls only.

Never imported by the server. Used by tests and by smoke_ws.py to prove the
reward channel pays 1.0 for a correct solution and 0 for doing nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qe_arena import edit_budget, sandbox  # noqa: E402
from qe_arena.tasks import MODULE, Task  # noqa: E402


def _expected(result: sandbox.CallResult):
    return {"raises": result.value} if result.kind == "exc" else result.value


def kill_tests(task: Task, *, wrong_expected: bool = False) -> list[dict]:
    """One killing test per mutant, from the stored hidden inputs."""
    p = task.private
    tests = []
    for m in p["mutants"]:
        outputs = sandbox.run_calls(m["source"], p["entry"], p["inputs"])
        index = next(i for i, (r, o) in enumerate(zip(p["reference"], outputs)) if not sandbox.compare(r, o))
        expected = _expected(p["reference"][index])
        if wrong_expected:
            expected = {"not": "the right value"}
        tests.append({"args": p["inputs"][index], "expected": expected})
    return tests


def solve_actions(task: Task) -> list[dict]:
    """Actions that earn reward 1.0."""
    if task.family == "qe_bugfix":
        if task.private.get("edit_budget") is not None:  # d4: a whole-file edit would be refused
            edits = [{"op": "edit", "path": MODULE, **e}
                     for e in edit_budget.minimal_fix(task.private["buggy"], task.private["original"])]
        else:
            edits = [{"op": "edit", "path": MODULE, "old": task.private["buggy"], "new": task.private["original"]}]
        return [{"op": "read", "path": "*"}, *edits, {"op": "run_tests"}, {"op": "submit", "answer": {}}]
    return [{"op": "read", "path": "*"}, {"op": "submit", "answer": {"tests": kill_tests(task)}}]
