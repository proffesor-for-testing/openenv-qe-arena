"""mathematics-or-formal-reasoning desk family (level 4): exact totals over a ledger of mixed number notations.

See ``desk_science`` for what a desk family is. The ledger is synthetic (exact arithmetic has no public-data
analogue worth pinning), but it is shaped like a real export: several hundred rows, corrections that supersede
earlier rows, void rows, and quantities written six different ways. Floating point cannot pass: totals must be
exact reduced fractions.
"""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction
from math import lcm

from . import Built, Family
from ._common import rows, write_csv

DOMAIN = "mathematics-or-formal-reasoning"
LEVEL = 4
HEADER = ["group", "entries", "total"]
GROUPS = ("alpha", "bravo", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper")
REPEATING = re.compile(r"^(-?)(\d+)\.(\d*)\((\d+)\)$")
FRACTION = re.compile(r"^(-?\d+)/(\d+)$")
MIXED = re.compile(r"^(-?)(\d+) (\d+)/(\d+)$")


def _parse(text: str, variant: str | None = None) -> Fraction:
    """A ledger quantity as an exact Fraction. ``variant`` names a wrong reading, for the traps."""
    text = text.strip()
    if m := REPEATING.match(text):
        sign, whole, fixed, cycle = m.groups()
        if variant == "repeating_truncated":
            value = Fraction(f"{whole}.{fixed}{cycle}")
        else:
            value = Fraction(int(whole)) + (Fraction(int(fixed or "0"), 10 ** len(fixed)) if fixed else 0) \
                + Fraction(int(cycle), (10 ** len(cycle) - 1) * 10 ** len(fixed))
        return -value if sign else value
    if m := MIXED.match(text):
        sign, whole, num, den = m.groups()
        if variant == "mixed_concat":
            value = Fraction(int(whole + num), int(den))
        else:
            value = int(whole) + Fraction(int(num), int(den))
        return -value if sign else value
    return Fraction(text)  # integers, decimals, a/b fractions and scientific notation


def _format(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _quantity(rng: random.Random) -> str:
    kind = rng.randrange(6)
    sign = "-" if rng.random() < 0.15 else ""
    if kind == 0:
        return f"{sign}{rng.randint(1, 500)}"
    if kind == 1:
        return f"{sign}{rng.randint(0, 99)}.{rng.randint(1, 999):03d}".rstrip("0")
    if kind == 2:
        den = rng.choice((3, 6, 7, 9, 11, 12, 13, 16, 21, 24))
        return f"{sign}{rng.randint(1, 4 * den)}/{den}"
    if kind == 3:
        den = rng.choice((3, 4, 5, 8, 12))
        return f"{sign}{rng.randint(1, 9)} {rng.randint(1, den - 1)}/{den}"
    if kind == 4:
        fixed = "" if rng.random() < 0.5 else str(rng.randint(0, 9))
        return f"{sign}{rng.randint(0, 20)}.{fixed}({rng.randint(1, 9)}{'' if rng.random() < 0.6 else rng.randint(0, 9)})"
    return f"{sign}{rng.randint(1, 999)}e-{rng.randint(1, 3)}"


def _ledger_build(level: int, rng: random.Random) -> Built:
    groups = rng.sample(GROUPS, rng.randint(5, 8))
    lines, next_id, issued = [], 1000, []
    for _ in range(rng.randint(650, 850)):
        roll = rng.random()
        if issued and roll < 0.08:
            entry = rng.choice(issued)  # a correction: same entry_id, it replaces the earlier row
        else:
            entry, next_id = f"E{next_id}", next_id + rng.randint(1, 3)
            issued.append(entry)
        status = "VOID" if rng.random() < 0.06 else rng.choice(("posted", "posted", "posted", "reconciled"))
        lines.append([entry, rng.choice(groups), _quantity(rng), status])
    files = {"ledger.csv": write_csv(["entry_id", "group", "quantity", "status"], lines),
             "request.json": json.dumps({"groups": sorted(groups)}, indent=2) + "\n"}
    statement = "\n".join([
        "Finance operations exported ledger.csv: one row per posting, in the order the postings were made. The "
        "auditors want exact per-group totals, with no rounding anywhere.",
        "",
        "Rules:",
        "- A row whose entry_id appeared earlier in the file is a correction: it replaces the earlier row for that "
        "entry_id entirely (including its group and status). Only the last row of each entry_id counts.",
        "- After applying corrections, ignore entries whose status is VOID. posted and reconciled both count.",
        "- quantity is written in one of several notations: an integer (12), a decimal (3.75), a fraction (7/12), "
        "a mixed number (2 3/4, meaning 2 + 3/4), a repeating decimal (0.1(6) means 0.1666..., the digits in "
        "parentheses repeat forever) or scientific notation (125e-3). Any of them may be negative.",
        "- Compute every total exactly. Write it as a reduced fraction numerator/denominator with a positive "
        "denominator (for example -17/6), or as a plain integer when the denominator is 1.",
        "- entries is the number of counted entries in the group.",
        "",
        "Write totals.csv with exactly this header line: " + ",".join(HEADER) + ", then one row for every group "
        "listed in request.json, in the order given there (a group with no counted entries has entries 0 and total "
        "0). The ledger is too long to read by eye, so use a script; Python's fractions module is available.",
    ])
    built = Built(files=files, statement=statement, output="totals.csv", reference=None)
    built.reference = rows(_ledger_solve(files, level, None))
    return built


def _ledger_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    groups = json.loads(files["request.json"])["groups"]
    latest: dict[str, dict[str, str]] = {}
    for row in rows(files["ledger.csv"]):
        if variant == "first_row_wins" and row["entry_id"] in latest:
            continue
        latest[row["entry_id"]] = row
    totals = {g: Fraction(0) for g in groups}
    counts = dict.fromkeys(groups, 0)
    floats = dict.fromkeys(groups, 0.0)
    for row in latest.values():
        if row["status"] == "VOID" and variant != "keep_void":
            continue
        value = _parse(row["quantity"], variant)
        totals[row["group"]] += value
        floats[row["group"]] += float(value)
        counts[row["group"]] += 1
    out = []
    for g in groups:
        if variant == "float_sum":
            total = repr(floats[g])
        elif variant == "unreduced":
            den = lcm(*(_parse(r["quantity"]).denominator for r in latest.values() if r["group"] == g), 1) * 2
            total = f"{totals[g].numerator * (den // totals[g].denominator)}/{den}"
        else:
            total = _format(totals[g])
        out.append([g, counts[g], total])
    return write_csv(HEADER, out)


def _ledger_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].strip() != ",".join(HEADER):
        return [False]
    out = rows(text)
    if len(out) != len(reference):
        return [False]
    results = []
    for got, ref in zip(out, reference):
        got = {k: (v or "").strip() for k, v in got.items() if k is not None}
        results += [got["group"] == ref["group"], got["entries"] == ref["entries"], got["total"] == ref["total"]]
    return results


FAMILIES = (
    Family("ledger", DOMAIN, _ledger_build, _ledger_check, _ledger_solve,
           {trap: (LEVEL,) for trap in ("first_row_wins", "keep_void", "repeating_truncated", "mixed_concat",
                                        "float_sum", "unreduced")},
           levels=(LEVEL,)),
)
