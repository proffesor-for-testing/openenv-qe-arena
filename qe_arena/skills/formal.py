"""mathematics-or-formal-reasoning families for qe_skill. (Named formal, not math, to keep the stdlib module.)"""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction

from . import Built, Family
from ._common import load_object

DOMAIN = "mathematics-or-formal-reasoning"

# M1 linear: an integer linear system solved exactly.
SIZES = {1: 2, 2: 3, 3: 4}
EQUATION = re.compile(r"^\s*(.+?)\s*=\s*(-?\d+)\s*$")
TERM = re.compile(r"([+-]?)\s*(\d*)\s*\*?\s*x(\d+)")
CANONICAL = re.compile(r"-?(0|[1-9]\d*)(/[1-9]\d*)?")


def _canonical(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _equation(coeffs: list[int], rhs: int) -> str:
    parts = []
    for i, c in enumerate(coeffs, 1):
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        mag = "" if abs(c) == 1 else f"{abs(c)}*"
        parts.append(f"{sign} {mag}x{i}")
    text = " ".join(parts) or "0"
    text = text[2:] if text.startswith("+ ") else "-" + text[2:] if text.startswith("- ") else text
    return f"{text} = {rhs}"


def _parse_system(text: str) -> tuple[list[list[Fraction]], list[Fraction], int]:
    eqs = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    parsed = []
    n = 0
    for line in eqs:
        match = EQUATION.match(line)
        lhs, rhs = match.group(1), int(match.group(2))
        terms = {}
        for sign, mag, idx in TERM.findall(lhs.replace(" ", "")):
            terms[int(idx)] = (-1 if sign == "-" else 1) * (int(mag) if mag else 1)
            n = max(n, int(idx))
        parsed.append((terms, rhs))
    a = [[Fraction(terms.get(j + 1, 0)) for j in range(n)] for terms, _ in parsed]
    b = [Fraction(rhs) for _, rhs in parsed]
    return a, b, n


def _eliminate(a: list[list[Fraction]], b: list[Fraction], n: int) -> tuple[str, list[Fraction]]:
    """("unique", x), ("none", []) or ("infinite", []) by exact Gauss-Jordan elimination."""
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    rank, pivots = 0, []
    for col in range(n):
        pivot = next((r for r in range(rank, len(m)) if m[r][col] != 0), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        m[rank] = [v / m[rank][col] for v in m[rank]]
        for r in range(len(m)):
            if r != rank and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * p for v, p in zip(m[r], m[rank])]
        pivots.append(col)
        rank += 1
    if any(all(v == 0 for v in row[:n]) and row[n] != 0 for row in m):
        return "none", []
    if rank < n:
        return "infinite", []
    x = [Fraction(0)] * n
    for r, col in enumerate(pivots):
        x[col] = m[r][n]
    return "unique", x


def _linear_build(level: int, rng: random.Random) -> Built:
    n = SIZES[level]
    kind = "unique" if level < 3 else rng.choice(("unique", "unique", "none", "infinite"))
    while True:
        a = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        if kind != "unique":  # make the last row a combination of two others; keep or break consistency
            i, j = rng.sample(range(n - 1), 2)
            p, q = rng.choice((1, 2, -1, 3)), rng.choice((1, -1, 2))
            a[-1] = [p * a[i][k] + q * a[j][k] for k in range(n)]
        b = [rng.randint(-30, 30) for _ in range(n)]
        if kind != "unique":
            b[-1] = p * b[i] + q * b[j] + (rng.choice((1, -1)) * rng.randint(1, 9) if kind == "none" else 0)
        status, x = _eliminate([[Fraction(v) for v in row] for row in a], [Fraction(v) for v in b], n)
        if status != kind or any(all(v == 0 for v in row) for row in a):
            continue
        if status == "unique" and not any(v.denominator > 1 for v in x):
            continue
        break
    order = list(range(n))
    rng.shuffle(order)
    system = "\n".join(_equation(a[r], b[r]) for r in order) + "\n"
    variables = ", ".join(f"x{i}" for i in range(1, n + 1))
    statement = "\n".join([
        f"system.txt holds a system of linear equations in {variables} with integer coefficients.",
        "Solve it exactly. Floating-point answers are wrong: give each value as a reduced fraction string such as "
        '"-7/3", or an integer string such as "4".',
        'Write solution.json as {"status": "unique", "x": ["<x1>", "<x2>", ...]} in variable order.'
        + (' If the system has no solution write {"status": "none"}; if it has infinitely many write '
           '{"status": "infinite"}.' if level == 3 else ""),
    ])
    built = Built(files={"system.txt": system}, statement=statement, output="solution.json", reference=None)
    built.reference = json.loads(_linear_solve(built.files, level, None))
    return built


def _linear_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    a, b, n = _parse_system(files["system.txt"])
    status, x = _eliminate(a, b, n)
    if variant == "singular_missed":  # applies only to a singular system; elsewhere it names no answer at all
        return json.dumps({"status": "unique", "x": ["0"] * n} if status != "unique" else {"status": "n/a"})
    if status != "unique":
        return json.dumps({"status": status})
    if variant == "float":
        return json.dumps({"status": "unique", "x": [f"{float(v):.6f}" for v in x]})
    if variant == "unreduced":  # Cramer's rule without reducing: every value over the common determinant
        det = abs(_det([row[:] for row in a]))
        return json.dumps({"status": "unique",
                           "x": [f"{int(v * det)}/{int(det)}" if det != 1 else str(int(v)) for v in x]})
    return json.dumps({"status": "unique", "x": [_canonical(v) for v in x]})


def _det(m: list[list[Fraction]]) -> Fraction:
    n, det = len(m), Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            det = -det
        det *= m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            m[r] = [v - f * p for v, p in zip(m[r], m[col])]
    return det


def _linear_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    if reference["status"] != "unique":
        return [out.get("status") == reference["status"]]
    x = out.get("x")
    results = [out.get("status") == "unique", isinstance(x, list) and len(x) == len(reference["x"])]
    if not all(results):
        return results
    return results + [isinstance(v, str) and CANONICAL.fullmatch(v) is not None and v == ref
                      for v, ref in zip(x, reference["x"])]


FAMILIES = (
    Family(name="linear", domain=DOMAIN, build=_linear_build, check=_linear_check, solve=_linear_solve,
           traps={"float": (1, 2, 3), "unreduced": (1, 2, 3), "singular_missed": (3,)}),
)
