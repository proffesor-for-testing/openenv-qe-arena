"""Edit budget for qe_bugfix-d4: a full rewrite does not fit, so the faults must be localised.

The budget depends on the shown file only: BUDGET_SHARE of its length, rounded up to a multiple of
BUDGET_QUANTUM. A budget computed from the faults (for example 1.5x their line diff, the
calibrated prototype) would tell the model how many characters the faulty lines hold. This one
tells it nothing it cannot already see. The share matches that prototype's median (0.39 of the
file). A task is served only if the minimal fix fits and a full rewrite does not.
"""

from __future__ import annotations

import difflib
import math

BUDGET_SHARE = 0.4
BUDGET_QUANTUM = 50  # characters


def _hunks(buggy: str, original: str):
    a, b = buggy.splitlines(keepends=True), original.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return a, b, [op for op in matcher.get_opcodes() if op[0] != "equal"]


def diff_bytes(buggy: str, original: str) -> int:
    """Characters on the changed lines of both sides: the size of the fix as a line diff."""
    a, b, hunks = _hunks(buggy, original)
    return sum(len("".join(a[i1:i2])) + len("".join(b[j1:j2])) for _, i1, i2, j1, j2 in hunks)


def minimal_fix(buggy: str, original: str) -> list[dict]:
    """Edits, applied in order, that turn ``buggy`` into ``original``.

    One edit per changed hunk, last hunk first, each widened by whole context lines until its
    ``old`` occurs exactly once in the file as it stands when that edit is applied.
    """
    a, b, hunks = _hunks(buggy, original)
    current = list(a)
    edits = []
    for _, i1, i2, j1, j2 in reversed(hunks):
        lo, hi = i1, i2
        while True:
            old = "".join(current[lo:hi])
            if old and "".join(current).count(old) == 1:
                break
            if lo > 0:
                lo -= 1
            elif hi < len(current):
                hi += 1
            else:
                break
        new_lines = current[lo:i1] + b[j1:j2] + current[i2:hi]
        edits.append({"old": "".join(current[lo:hi]), "new": "".join(new_lines)})
        current[lo:hi] = new_lines
    if "".join(current) != original:  # pragma: no cover - defensive invariant
        raise AssertionError("minimal_fix does not restore the original")
    return edits


def edit_cost(edit: dict) -> int:
    return len(edit["old"]) + len(edit["new"])


def budget_for(shown: str) -> int:
    """The budget for a file: a function of the shown text only, never of the faults."""
    raw = math.ceil(BUDGET_SHARE * len(shown))
    return -(-raw // BUDGET_QUANTUM) * BUDGET_QUANTUM


def plan(buggy: str, original: str) -> tuple[int, list[dict]] | None:
    """(budget, minimal fix) when the minimal fix fits and a full rewrite does not, else None."""
    budget = budget_for(buggy)
    fix = minimal_fix(buggy, original)
    if sum(edit_cost(e) for e in fix) > budget or len(buggy) + len(original) <= budget:
        return None
    return budget, fix
