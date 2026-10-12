"""Reference solutions built from private task data, for verifier controls only.

Never imported by the server. Used by tests and by smoke_ws.py to prove the
reward channel pays 1.0 for a correct solution and 0 for doing nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qe_arena import sandbox  # noqa: E402
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
        return [
            {"op": "read", "path": "*"},
            {"op": "edit", "path": MODULE, "old": task.private["buggy"], "new": task.private["original"]},
            {"op": "run_tests"},
            {"op": "submit", "answer": {}},
        ]
    if task.family == "qe_flaky":
        p = task.private
        actions = [{"op": "read", "path": "*"}, {"op": "run_tests"}]
        if p["repair"]:
            actions.append({"op": "edit", "path": MODULE, **p["repair"]})
        answer = {"test": p["victim"]} if p["victim"] is None or task.difficulty != 2 else \
            {"test": p["victim"], "cause": p["cause"]}
        return actions + [{"op": "submit", "answer": answer}]
    return [{"op": "read", "path": "*"}, {"op": "submit", "answer": {"tests": kill_tests(task)}}]
