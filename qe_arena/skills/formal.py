"""mathematics-or-formal-reasoning families for qe_skill. (Named formal, not math, to keep the stdlib module.)"""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction
from itertools import permutations
from math import factorial, gcd, prod

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
        + (' If the system has no solution, or infinitely many, write an object whose only key is "status", set '
           'to "none" or "infinite" respectively.' if level == 3 else ""),
    ])
    built = Built(files={"system.txt": system}, statement=statement, output="solution.json", reference=None)
    built.reference = json.loads(_linear_solve(built.files, level, None))
    return built


def _linear_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    a, b, n = _parse_system(files["system.txt"])
    status, x = _eliminate(a, b, n)
    if variant == "singular_missed":  # applies only to a singular system; elsewhere it names no answer at all
        return json.dumps({"status": "unique", "x": ["0"] * n} if status != "unique" else {"status": "n/a"})
    if status != "unique":  # the fraction-format traps name no answer for a singular system
        return json.dumps({"status": "n/a" if variant in ("float", "unreduced") else status})
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


def _exact_int(value: object, reference: int) -> bool:
    return type(value) is int and value == reference


# M2 counting: arrangements of letters in a row under constraints, counted by brute force.
COUNT_SIZES = {1: 5, 2: 6, 3: 7}


def _violates(arrangement: tuple[str, ...], c: dict) -> bool:
    if c["type"] == "not_at":
        return arrangement[c["position"] - 1] == c["item"]
    x, y = c["items"]
    if c["type"] == "not_adjacent":
        return any({p, q} == {x, y} for p, q in zip(arrangement, arrangement[1:]))
    return arrangement.index(x) > arrangement.index(y)  # "before": x left of y; both letters appear once


def _constraint_text(c: dict) -> str:
    if c["type"] == "not_at":
        return f"no {c['item']} in position {c['position']}"
    x, y = c["items"]
    return f"no {x} directly next to a {y}" if c["type"] == "not_adjacent" else f"{x} somewhere left of {y}"


def _violation_masks(items: list[str], constraints: list[dict]) -> list[int]:
    """One bitmask per distinct arrangement: bit i set when the arrangement breaks constraint i."""
    return [sum(1 << i for i, c in enumerate(constraints) if _violates(arr, c))
            for arr in sorted(set(permutations(items)))]


def _counting_build(level: int, rng: random.Random) -> Built:
    n = COUNT_SIZES[level]
    while True:
        doubles = 0 if level == 1 else rng.choice((1, 2)) if level == 3 else 1
        letters = [chr(65 + i) for i in range(n - doubles)]
        items = sorted(letters + rng.sample(letters, doubles))
        singles = [x for x in letters if items.count(x) == 1]
        constraints: list[dict] = []
        while len(constraints) < level:
            kind = rng.choice(("not_at", "not_adjacent", "before"))
            if kind == "not_at":
                c = {"type": kind, "item": rng.choice(letters), "position": rng.randint(1, n)}
            else:
                c = {"type": kind, "items": rng.sample(singles if kind == "before" else letters, 2)}
            if all(_constraint_text(c) != _constraint_text(o) for o in constraints):
                constraints.append(c)
        masks = _violation_masks(items, constraints)
        count = masks.count(0)
        full = (1 << level) - 1
        binding = all(sum(1 for m in masks if m & (full ^ (1 << i)) == 0) != count for i in range(level))
        overlap = len(masks) - sum(bin(m).count("1") for m in masks) != count
        if count > 0 and binding and (level == 1 or overlap):
            break
    for c in constraints:
        c["text"] = _constraint_text(c)
    problem = {"items": items, "constraints": constraints}
    statement = "\n".join([
        "problem.json lists items to arrange in a row (place every entry of items exactly once, so a letter listed "
        "twice appears twice) and constraints on the row. Count the distinct arrangements that satisfy every "
        "constraint.",
        "Copies of a repeated letter are identical: two arrangements are the same when every position holds the "
        "same letter. Positions are numbered from 1 at the left.",
        'Constraint types: "not_at" means no copy of item stands at that position; "not_adjacent" with items [X, Y] '
        'means no X stands directly next to a Y (either side); "before" with items [X, Y] means X stands somewhere '
        "left of Y. Each constraint also carries a plain-English text.",
        'Write answer.json as {"count": <integer>}, the count as a JSON integer.',
    ])
    built = Built(files={"problem.json": json.dumps(problem, indent=2) + "\n"}, statement=statement,
                  output="answer.json", reference=None)
    built.reference = json.loads(_counting_solve(built.files, level, None))
    return built


def _counting_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    problem = json.loads(files["problem.json"])
    items, constraints = problem["items"], problem["constraints"]
    if variant == "drop_a_constraint":
        constraints = constraints[:-1]
    masks = _violation_masks(items, constraints)
    count = masks.count(0)
    if variant == "ignore_overlap":  # total minus each constraint's violations, intersections never added back
        count = len(masks) - sum(bin(m).count("1") for m in masks)
    if variant == "order_matters_twice":  # copies treated as distinct; applies only when a letter repeats
        copies = prod(factorial(items.count(x)) for x in set(items))
        return json.dumps({"count": count * copies} if copies > 1 else {"status": "n/a"})
    return json.dumps({"count": count})


def _counting_check(reference: dict, text: str) -> list[bool]:
    return [_exact_int(load_object(text).get("count"), reference["count"])]


# M3 congruences: x = r_i (mod m_i), solved by combining one congruence at a time.
def _combine(pairs: list[tuple[int, int]], strict: bool = True) -> tuple[int, int] | None:
    """(least non-negative x, lcm) of the system, or None when it is inconsistent (``strict``)."""
    x, m = 0, 1
    for r, mod in pairs:
        g = gcd(m, mod)
        if strict and (r - x) % g:
            return None
        t = ((r - x) // g * pow(m // g, -1, mod // g)) % (mod // g) if mod // g > 1 else 0
        x, m = x + m * t, m * mod // g
        x %= m
    return x, m


def _congruences_build(level: int, rng: random.Random) -> Built:
    count = 2 if level == 1 else 3 if level == 2 else rng.choice((3, 4))
    inconsistent = level == 3 and rng.random() < 1 / 3
    while True:
        moduli = rng.sample(range(3, 31 if level == 1 else 46), count)
        lcm = _combine([(0, m) for m in moduli])[1]
        coprime = lcm == prod(moduli)
        if (level == 1) != coprime or lcm > 20000 or any(a % b == 0 for a in moduli for b in moduli if a != b):
            continue
        x0 = rng.randrange(lcm)
        pairs = [(x0 % m, m) for m in moduli]
        if inconsistent:
            i = rng.randrange(count)
            pairs[i] = (rng.randrange(moduli[i]), moduli[i])
            if _combine(pairs) is not None:
                continue
        break
    system = {"congruences": [{"remainder": r, "modulus": m} for r, m in pairs]}
    statement = "\n".join([
        "system.json lists congruences x = remainder (mod modulus). Find every integer x that satisfies all of them.",
        'Write solution.json as {"status": "solved", "x": <least non-negative solution>, "modulus": <M>}, where M '
        "is the smallest positive integer such that the solutions are exactly the integers congruent to x mod M. "
        "Both values are JSON integers."
        + (' If no integer satisfies every congruence, write an object whose only key is "status", set to "none".'
           if level == 3 else ""),
    ])
    built = Built(files={"system.json": json.dumps(system, indent=2) + "\n"}, statement=statement,
                  output="solution.json", reference=None)
    built.reference = json.loads(_congruences_solve(built.files, level, None))
    return built


def _congruences_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    pairs = [(c["remainder"], c["modulus"]) for c in json.loads(files["system.json"])["congruences"]]
    solved = _combine(pairs)
    if variant == "inconsistent_missed":  # applies only to an inconsistent system: drop the congruence that clashes
        rest = next((r for i in range(len(pairs)) if (r := _combine(pairs[:i] + pairs[i + 1:]))), None)
        return json.dumps({"status": "solved", "x": rest[0], "modulus": rest[1]} if solved is None and rest
                          else {"status": "n/a"})
    if variant == "assume_coprime":  # the coprime formula: no consistency check, modulus the product
        x, _ = _combine(pairs, strict=False) if solved is None else solved
        return json.dumps({"status": "solved", "x": x, "modulus": prod(m for _, m in pairs)})
    if solved is None:
        return json.dumps({"status": "n/a"} if variant == "not_least" else {"status": "none"})
    x, m = solved
    if variant == "not_least":
        x += m
    return json.dumps({"status": "solved", "x": x, "modulus": m})


def _congruences_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    if reference["status"] == "none":
        return [out.get("status") == "none"]
    return [out.get("status") == "solved", _exact_int(out.get("x"), reference["x"]),
            _exact_int(out.get("modulus"), reference["modulus"])]


# M4 falsify: the smallest counterexample to a claim over an integer range, or "holds".
GRID = 18  # claims are scanned on 0..GRID-1 per variable while choosing ranges


def _poly(terms: list[dict], point: dict[str, int]) -> int:
    return sum(t["coef"] * prod(point[v] ** p for v, p in t.get("powers", {}).items()) for t in terms)


def _poly_text(terms: list[dict]) -> str:
    parts = []
    for t in terms:
        mono = "*".join(v if p == 1 else f"{v}^{p}" for v, p in t.get("powers", {}).items())
        c = t["coef"]
        body = mono if mono and abs(c) == 1 else f"{abs(c)}*{mono}" if mono else str(abs(c))
        parts.append(("- " if c < 0 else "+ ") + body)
    text = " ".join(parts)
    return text[2:] if text.startswith("+ ") else "-" + text[2:]


def _is_prime(k: int) -> bool:
    return k >= 2 and all(k % d for d in range(2, int(k ** 0.5) + 1))


def _holds(claim: dict, point: dict[str, int]) -> bool:
    kind = claim["kind"]
    if kind == "at_most":
        return _poly(claim["left"], point) <= _poly(claim["right"], point)
    if kind == "prime":
        return _is_prime(_poly(claim["expr"], point))
    if kind == "not_divisible":
        return _poly(claim["expr"], point) % claim["divisor"] != 0
    return gcd(_poly(claim["left"], point), _poly(claim["right"], point)) == 1  # coprime


def _claim_text(claim: dict) -> str:
    kind = claim["kind"]
    if kind == "at_most":
        return f"{_poly_text(claim['left'])} <= {_poly_text(claim['right'])}"
    if kind == "prime":
        return f"{_poly_text(claim['expr'])} is prime"
    if kind == "not_divisible":
        return f"{_poly_text(claim['expr'])} is not divisible by {claim['divisor']}"
    return f"gcd({_poly_text(claim['left'])}, {_poly_text(claim['right'])}) = 1"


def _search(claim: dict, ranges: dict[str, list[int]], *, b_major: bool = False,
            python_range: bool = False) -> tuple[int, ...] | None:
    """The first counterexample in lexicographic order (or b before a, or range(hi) per variable), else None."""
    axes = [range(0, hi) if python_range else range(lo, hi + 1) for lo, hi in ranges.values()]
    names = list(ranges)
    if len(axes) == 1:
        return next(((n,) for n in axes[0] if not _holds(claim, {names[0]: n})), None)
    order = ((a, b) for b in axes[1] for a in axes[0]) if b_major else ((a, b) for a in axes[0] for b in axes[1])
    return next(((a, b) for a, b in order if not _holds(claim, dict(zip(names, (a, b))))), None)


def _term(rng: random.Random, names: list[str], degree: int) -> dict:
    powers = {}
    for _ in range(degree):
        v = rng.choice(names)
        powers[v] = powers.get(v, 0) + 1
    return {"coef": rng.choice((1, 1, 2, 3, -1, -2)), "powers": dict(sorted(powers.items()))}


def _random_claim(rng: random.Random, names: list[str]) -> dict:
    one = len(names) == 1
    kinds = ("at_most", "prime", "not_divisible") if one else ("at_most", "at_most", "not_divisible", "coprime")
    kind = rng.choice(kinds)
    const = {"coef": rng.randint(1, 41)}
    if kind == "prime":
        return {"kind": kind, "expr": [{"coef": 1, "powers": {"n": 2}}, {"coef": rng.randint(-9, 9) or 1,
                                       "powers": {"n": 1}}, {"coef": rng.choice(range(1, 42, 2))}]}
    if kind == "not_divisible":
        return {"kind": kind, "expr": [_term(rng, names, 2), _term(rng, names, rng.randint(1, 2)), const],
                "divisor": rng.randint(5, 13)}
    if kind == "coprime":
        return {"kind": kind, "left": [_term(rng, names, 1), _term(rng, names, 1), const],
                "right": [_term(rng, names, 2), {"coef": rng.randint(1, 9)}]}
    left = [{"coef": rng.randint(1, 3), "powers": _term(rng, names, rng.randint(2, 3))["powers"]}]
    right = [{"coef": rng.randint(2, 12), "powers": {v: 1}} for v in rng.sample(names, len(names))] + [const]
    return {"kind": kind, "left": left, "right": right}


def _random_ranges(rng: random.Random, names: list[str]) -> dict[str, list[int]]:
    ranges = {}
    for v in names:
        lo = rng.randint(1, 8)
        ranges[v] = [lo, min(GRID - 1, lo + rng.randint(3, 9 if len(names) == 2 else 9))]
    return ranges


def _falsify_ok(claim: dict, ranges: dict[str, list[int]], level: int, want_holds: bool) -> bool:
    answer = _search(claim, ranges)
    if (answer is None) != want_holds or _search(claim, ranges, python_range=True) == answer:
        return False
    if want_holds or level == 1:
        return True
    if _search(claim, ranges, b_major=True) == answer:
        return False
    lo, hi = ranges["a"]
    return level == 2 or answer[0] - lo >= (hi - lo) * 2 / 3  # d3: the first counterexample sits late in a


def _falsify_build(level: int, rng: random.Random) -> Built:
    names = ["n"] if level == 1 else ["a", "b"]
    want_holds = level == 3 and rng.random() < 1 / 3
    while True:
        claim = _random_claim(rng, names)
        ranges = next((r for r in (_random_ranges(rng, names) for _ in range(12))
                       if _falsify_ok(claim, r, level, want_holds)), None)
        if ranges is not None:
            break
    bounds = " and ".join(f"{lo} <= {v} <= {hi}" for v, (lo, hi) in ranges.items())
    claim["text"] = f"for all integers {', '.join(names)} with {bounds}: {_claim_text(claim)}"
    document = {"variables": names, "ranges": ranges, "claim": claim}
    one = level == 1
    statement = "\n".join([
        "claim.json states a claim about integers over a search range and asks whether it holds there. Both range "
        "bounds are inclusive.",
        "Each expression is a list of terms; a term is coef times the product of each variable raised to its power "
        '(a term with no powers is a constant). Claim kinds: "at_most" means left <= right; "prime" means expr is a '
        'prime number (an integer of at least 2 whose only positive divisors are 1 and itself); "not_divisible" '
        'means expr is not a multiple of divisor (0 is a multiple of everything); "coprime" means the greatest '
        "common divisor of left and right is 1 (taken on absolute values, so gcd(0, k) = |k|). The text field "
        "restates the claim.",
        "Find the smallest counterexample in the range: " + (
            "the smallest n for which the claim is false." if one else
            "order points by a first and then by b, so (a, b) comes before (a', b') when a < a', or a = a' and "
            "b < b'."),
        'Write answer.json as {"result": "counterexample", ' + ('"n": <n>}' if one else '"a": <a>, "b": <b>}')
        + ' with JSON integers. If the claim is true at every point of the range, write an object whose only key is '
        '"result", set to "holds".',
    ])
    built = Built(files={"claim.json": json.dumps(document, indent=2) + "\n"}, statement=statement,
                  output="answer.json", reference=None)
    built.reference = json.loads(_falsify_solve(built.files, level, None))
    return built


def _falsify_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    document = json.loads(files["claim.json"])
    claim, ranges, names = document["claim"], document["ranges"], document["variables"]
    answer = _search(claim, ranges, b_major=variant == "first_found_not_smallest",
                     python_range=variant == "off_by_one_range")
    if variant == "always_counterexample":  # never says holds; applies only where the claim does hold
        if _search(claim, ranges) is not None:
            return json.dumps({"result": "n/a"})
        answer = tuple(lo for lo, _ in ranges.values())
    if variant == "first_found_not_smallest" and answer is None:
        return json.dumps({"result": "n/a"})
    if answer is None:
        return json.dumps({"result": "holds"})
    return json.dumps({"result": "counterexample", **dict(zip(names, answer))})


def _falsify_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    if reference["result"] == "holds":
        return [out.get("result") == "holds"]
    return [out.get("result") == "counterexample"] + [_exact_int(out.get(k), v)
                                                      for k, v in reference.items() if k != "result"]


FAMILIES = (
    Family(name="linear", domain=DOMAIN, build=_linear_build, check=_linear_check, solve=_linear_solve,
           traps={"float": (1, 2, 3), "unreduced": (1, 2, 3), "singular_missed": (3,)}),
    Family(name="counting", domain=DOMAIN, build=_counting_build, check=_counting_check, solve=_counting_solve,
           traps={"ignore_overlap": (2, 3), "drop_a_constraint": (1, 2, 3), "order_matters_twice": (2, 3)}),
    Family(name="congruences", domain=DOMAIN, build=_congruences_build, check=_congruences_check,
           solve=_congruences_solve,
           traps={"assume_coprime": (2, 3), "not_least": (1, 2, 3), "inconsistent_missed": (3,)}),
    Family(name="falsify", domain=DOMAIN, build=_falsify_build, check=_falsify_check, solve=_falsify_solve,
           traps={"off_by_one_range": (1, 2, 3), "first_found_not_smallest": (2, 3),
                  "always_counterexample": (3,)}),
)
