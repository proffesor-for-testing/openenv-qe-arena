"""Mutation engine: every site is one exact, re-parseable, formatting-preserving replacement."""

from __future__ import annotations

import ast
import random
from collections import Counter

import pytest

from qe_arena import mutation, sandbox
from qe_arena.mutation import Site

SAMPLE = '''\
def helper(xs, lo, hi):
    # count values inside [lo, hi)
    total = 0  # running count
    for i in range(len(xs)):
        if xs[i] is None:
            continue
        if lo <= xs[i] and xs[i] < hi:
            total += 1
        elif not xs[i] in (0, 1):
            break
    return total


def sample(xs, k):
    """Docstring with < and + and 7 that must not be mutated."""
    if k == 0 or not xs:
        return False
    window = xs[1:k]
    best = min(window) if window else max(xs)
    label = "n=" + str(k)   # str concatenation is never swapped
    if len(label) >= 9:
        return True
    return helper(xs, best, best * 2 + 3) > k - 1
'''


def _offsets(source: str, site: Site) -> tuple[int, int]:
    text = mutation._Text(source)
    return text.offset(site.lineno, site.col), text.offset(site.end_lineno, site.end_col)


def _ints(seed: int, n: int) -> list[list]:
    rng = random.Random(seed)
    return [[[rng.choice([None, *range(-3, 12)]) for _ in range(rng.randint(0, 9))], rng.randint(-2, 8)]
            for _ in range(n)]


# --------------------------------------------------------------------------- enumerate_sites

def test_sample_covers_every_operator():
    counts = Counter(s.operator for s in mutation.enumerate_sites(SAMPLE))
    assert set(counts) == set(mutation.OPERATORS), counts


def test_every_site_reparses_and_changes_only_its_span():
    for site in mutation.enumerate_sites(SAMPLE):
        mutated = mutation.apply(SAMPLE, [site])
        start, end = _offsets(SAMPLE, site)
        assert SAMPLE[start:end] == site.original
        assert mutated[:start] == SAMPLE[:start]
        assert mutated[start:start + len(site.replacement)] == site.replacement
        assert mutated[start + len(site.replacement):] == SAMPLE[end:]
        assert ast.dump(ast.parse(mutated)) != ast.dump(ast.parse(SAMPLE)), site


def test_comments_and_docstring_survive_every_mutation():
    keep = ["# count values inside [lo, hi)", "# running count", '"""Docstring with < and + and 7',
            "# str concatenation is never swapped"]
    for site in mutation.enumerate_sites(SAMPLE):
        mutated = mutation.apply(SAMPLE, [site])
        for fragment in keep:
            assert fragment in mutated


def test_no_site_inside_docstring_or_string_concatenation():
    doc_line = SAMPLE.splitlines().index('    """Docstring with < and + and 7 that must not be mutated."""') + 1
    label_line = next(i for i, l in enumerate(SAMPLE.splitlines(), 1) if 'label = "n="' in l)
    sites = mutation.enumerate_sites(SAMPLE)
    assert not [s for s in sites if s.lineno == doc_line]
    assert not [s for s in sites if s.lineno == label_line and s.operator == "arith_swap"]


def test_enumeration_is_deterministic_and_sorted():
    first = mutation.enumerate_sites(SAMPLE)
    second = mutation.enumerate_sites(SAMPLE)
    assert first == second
    keys = [(s.lineno, s.col, s.end_lineno, s.end_col, s.operator, s.replacement) for s in first]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def _only(source: str, operator: str) -> list[tuple[str, str]]:
    return [(s.original, s.replacement) for s in mutation.enumerate_sites(source) if s.operator == operator]


@pytest.mark.parametrize("source, operator, expected", [
    ("def f(a, b):\n    return a < b\n", "cmp_boundary", [("<", "<=")]),
    ("def f(a, b):\n    return a >= b\n", "cmp_boundary", [(">=", ">")]),
    ("def f(a, b):\n    return a != b\n", "cmp_negate", [("!=", "==")]),
    ("def f(a, b):\n    return a not in b\n", "cmp_negate", [("not in", "in")]),
    ("def f(a, b):\n    return a is not b\n", "cmp_negate", [("is not", "is")]),
    ("def f(a, b):\n    return (a) in (b)\n", "cmp_negate", [("in", "not in")]),
    ("def f(a, b):\n    return a // b\n", "arith_swap", [("//", "*")]),
    ("def f(a, b):\n    return a % b\n", "arith_swap", []),
    ("def f(a):\n    return 'x' + a\n", "arith_swap", []),
    ("def f(a):\n    return f'{a}' + a\n", "arith_swap", []),
    ("def f(a):\n    return a and not a or a\n", "bool_swap", [("and", "or"), ("or", "and")]),
    ("def f(a):\n    return not(a)\n", "not_drop", [("not", "")]),
    ("def f(a):\n    a -= 2\n    return a\n", "augassign_swap", [("-=", "+=")]),
    ("def f(a):\n    a *= 2\n    return a\n", "augassign_swap", []),
    ("def f(a):\n    return max(a, 1)\n", "min_max_swap", [("max", "min")]),
    ("def f(a):\n    return a.max()\n", "min_max_swap", []),
    ("def f(a):\n    return False\n", "return_flip", [("False", "True")]),
    ("def f(a):\n    return a == True\n", "return_flip", []),
    ("def f(a):\n    return a[i:]\n", "slice_shift", [("i", "i + 1"), ("i", "i - 1")]),
    ("def f(a):\n    return a[:n - 1]\n", "slice_shift", [("n - 1", "n - 1 + 1"), ("n - 1", "n - 1 - 1")]),
    ("def f(a):\n    return a[:x if c else y]\n", "slice_shift",
     [("x if c else y", "(x if c else y) + 1"), ("x if c else y", "(x if c else y) - 1")]),
    ("def f(a):\n    return a[1:-1]\n", "slice_shift", []),
    ("def f(n):\n    return list(range(n))\n", "range_bound", [("n", "n + 1"), ("n", "n - 1")]),
    ("def f(n):\n    return list(range(2, n, 3))\n", "range_bound", [("n", "n + 1"), ("n", "n - 1")]),
    ("def f(n):\n    return list(range(10))\n", "range_bound", []),
])
def test_operator_specifics(source, operator, expected):
    assert sorted(_only(source, operator)) == sorted(expected)


def test_const_skips_bool_and_range_step_and_parenthesizes_where_needed():
    source = "def f(n):\n    a = list(range(0, n, 2))\n    b = True\n    return 0 ** n, a, b\n"
    pairs = _only(source, "const_off_by_one")
    assert ("2", "3") not in pairs and ("2", "1") not in pairs
    assert ("0", "(-1)") in pairs and ("0", "1") in pairs  # 0 ** n -> (-1) ** n, not -(1 ** n)
    assert ("0", "-1") in pairs  # range start
    mutated = mutation.apply(source, [s for s in mutation.enumerate_sites(source)
                                      if s.operator == "const_off_by_one" and s.replacement == "(-1)"])
    assert "return (-1) ** n" in mutated


def test_const_keeps_literal_spelling_out_of_the_rest_of_the_file():
    source = "def f(n):\n    return n + 0x10  # hex\n"
    pairs = _only(source, "const_off_by_one")
    assert pairs == [("0x10", "15"), ("0x10", "17")]
    assert mutation.apply(source, [mutation.enumerate_sites(source)[-1]]) == "def f(n):\n    return n + 17  # hex\n"


def test_fstring_contents_are_never_mutated():
    source = "def f(a):\n    return f'{a + 1}' if a < 2 else ''\n"
    sites = mutation.enumerate_sites(source)
    assert [(s.operator, s.original) for s in sites] == [("cmp_boundary", "<"), ("const_off_by_one", "2"),
                                                          ("const_off_by_one", "2")]


def test_non_ascii_columns_are_character_offsets():
    source = "def f(a):\n    s = 'é—ü'; return a < 3  # naïve\n"
    for site in mutation.enumerate_sites(source):
        mutated = mutation.apply(source, [site])
        ast.parse(mutated)
        assert "'é—ü'" in mutated and "# naïve" in mutated
    cmp = next(s for s in mutation.enumerate_sites(source) if s.operator == "cmp_boundary")
    assert source.splitlines()[1][cmp.col:cmp.end_col] == "<"


def test_crlf_source_is_spliced_exactly():
    source = "def f(a, b):\r\n    # c\r\n    return a < b\r\n"
    site = mutation.enumerate_sites(source)[0]
    assert mutation.apply(source, [site]) == "def f(a, b):\r\n    # c\r\n    return a <= b\r\n"


def test_mutants_change_behaviour_on_sample():
    """Most single mutants of SAMPLE are behaviourally distinct (the engine is not cosmetic)."""
    calls = _ints(7, 120)
    reference = sandbox.run_calls(SAMPLE, "sample", calls)
    assert all(r.kind in ("ok", "exc") for r in reference)
    sites = mutation.enumerate_sites(SAMPLE)
    killed_by_operator: Counter = Counter()
    for site in sites:
        outputs = sandbox.run_calls(mutation.apply(SAMPLE, [site]), "sample", calls)
        killed_by_operator[site.operator] += any(not sandbox.compare(r, o) for r, o in zip(reference, outputs))
    # The `len(label) >= 9` branch is unreachable for these k, so its mutants are equivalent.
    assert {op for op, n in killed_by_operator.items() if n} == set(mutation.OPERATORS)
    assert sum(killed_by_operator.values()) / len(sites) >= 0.6


# --------------------------------------------------------------------------- apply

def test_apply_overlap_raises():
    sites = mutation.enumerate_sites("def f(a):\n    return a[:n - 1]\n")
    shift = next(s for s in sites if s.operator == "slice_shift")
    inner = next(s for s in sites if s.operator == "arith_swap")
    with pytest.raises(ValueError, match="overlap"):
        mutation.apply("def f(a):\n    return a[:n - 1]\n", [shift, inner])


def test_apply_same_span_twice_raises():
    source = "def f(a):\n    return a + 1\n"
    first, second = [s for s in mutation.enumerate_sites(source) if s.operator == "const_off_by_one"]
    with pytest.raises(ValueError):
        mutation.apply(source, [first, second])
    with pytest.raises(ValueError):
        mutation.apply(source, [first, first])


def test_apply_stale_site_raises():
    site = mutation.enumerate_sites("def f(a):\n    return a < 1\n")[0]
    with pytest.raises(ValueError, match="does not match"):
        mutation.apply("def f(a):\n    return a > 1\n", [site])


def test_apply_multiple_sites_is_order_independent():
    sites = mutation.enumerate_sites(SAMPLE)
    chosen = [sites[0], sites[len(sites) // 2], sites[-1]]
    a = mutation.apply(SAMPLE, chosen)
    b = mutation.apply(SAMPLE, list(reversed(chosen)))
    assert a == b
    ast.parse(a)
    assert mutation.apply(SAMPLE, []) == SAMPLE


# --------------------------------------------------------------------------- unified_diff

def test_unified_diff_shape():
    site = next(s for s in mutation.enumerate_sites(SAMPLE) if s.operator == "augassign_swap")
    diff = mutation.unified_diff(SAMPLE, mutation.apply(SAMPLE, [site]), "module.py")
    lines = diff.splitlines()
    assert lines[0] == "--- a/module.py" and lines[1] == "+++ b/module.py"
    assert lines[2].startswith("@@")
    assert [l for l in lines if l.startswith("-") and not l.startswith("---")] == ["-            total += 1"]
    assert [l for l in lines if l.startswith("+") and not l.startswith("+++")] == ["+            total -= 1"]
    assert len(lines) == 3 + 2 + 2 + 2  # headers, hunk header, 2 context lines each side, -/+
    assert mutation.unified_diff("a = 1", "a = 2", "x.py").endswith("+a = 2\n")


# --------------------------------------------------------------------------- rename_locals

RENAME_SAMPLES = {
    "closures": ('''\
LIMIT = 3


def outer(xs, n):
    acc = []  # collected

    def push(v):
        acc.append(v * 2)
    for item in xs:
        push(item)
    total = sum(acc) + LIMIT
    evens = [y for y in acc if y % 4 == 0]
    return [total, evens, (lambda q: q + total)(n), {k: k * n for k in evens}]
''', "outer", lambda rng: [[rng.randint(-5, 9) for _ in range(rng.randint(0, 7))], rng.randint(-3, 3)]),
    "walrus_try": ('''\
def f(xs, n):
    key = (lambda v: -v) if n % 2 else (lambda v: v)
    ordered = sorted(xs, key=key)
    out = []
    i = 0
    while (j := i + 1) <= len(ordered):
        out.append(ordered[i])
        i = j
    try:
        bad = ordered[n]
    except IndexError as err:
        bad = None
    del i
    return [out, bad, f"{len(out)}"]
''', "f", lambda rng: [[rng.randint(-5, 9) for _ in range(rng.randint(0, 7))], rng.randint(-3, 8)]),
    "percentile": ('''\
def percentile(sorted_values, p):
    n = len(sorted_values)
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1 if f < n - 1 else f
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def summary(times):
    if not times:
        return {key: 0 for key in ("min", "max", "p50")}
    ordered = sorted(times)
    result = {"min": ordered[0], "max": ordered[-1]}
    for p in (25, 50, 75):
        result["p" + str(p)] = round(percentile(ordered, p), 2)
    return result
''', "summary", lambda rng: [[rng.randint(1, 500) for _ in range(rng.randint(0, 9))]]),
    "nested_loops": ('''\
import math


def g(grid):
    best = -math.inf
    for row in grid:
        run = 0
        for cell in row:
            if cell > 0:
                run += cell
            else:
                best = max(best, run)
                run = 0
        best = max(best, run)
    seen = set()
    for row in grid:
        for cell in row:
            seen.add(cell)
    return [best if best != -math.inf else None, sorted(seen)]
''', "g", lambda rng: [[[rng.randint(-3, 5) for _ in range(rng.randint(0, 4))] for _ in range(rng.randint(0, 4))]]),
}


def _exec_calls(source: str, entry: str, calls: list[list]) -> list:
    ns: dict = {}
    exec(compile(source, "<sample>", "exec"), ns)
    out = []
    for args in calls:
        try:
            out.append(("ok", repr(ns[entry](*args))))
        except Exception as exc:  # noqa: BLE001 - behaviour comparison only
            out.append(("exc", type(exc).__name__))
    return out


@pytest.mark.parametrize("name", sorted(RENAME_SAMPLES))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_rename_locals_is_semantically_identical(name, seed):
    source, entry, gen = RENAME_SAMPLES[name]
    renamed = mutation.rename_locals(source, random.Random(seed))
    rng = random.Random(1000 + seed)
    calls = [gen(rng) for _ in range(200)]
    assert _exec_calls(renamed, entry, calls) == _exec_calls(source, entry, calls)


def test_rename_locals_matches_through_the_sandbox():
    source, entry, gen = RENAME_SAMPLES["percentile"]
    renamed = mutation.rename_locals(source, random.Random(9))
    rng = random.Random(5)
    calls = [gen(rng) for _ in range(64)]
    reference = sandbox.run_calls(source, entry, calls)
    outputs = sandbox.run_calls(renamed, entry, calls)
    assert all(r.kind == "ok" for r in reference)
    assert all(sandbox.compare(r, o) for r, o in zip(reference, outputs))


def test_rename_locals_actually_renames_and_keeps_the_interface():
    source = RENAME_SAMPLES["closures"][0]
    renamed = mutation.rename_locals(source, random.Random(3))
    assert renamed != source
    tree = ast.parse(renamed)
    funcs = {n.name: [a.arg for a in n.args.args] for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert funcs == {"outer": ["xs", "n"], "push": ["v"]}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "LIMIT" in names and "acc" not in names and "item" not in names and "total" not in names
    assert ".append(" in renamed and "lambda q: q +" in renamed
    assert "# collected" in renamed
    assert renamed.count("\n") == source.count("\n")


def test_rename_locals_leaves_keywords_exception_names_and_fstring_names():
    source = RENAME_SAMPLES["walrus_try"][0]
    renamed = mutation.rename_locals(source, random.Random(0))
    assert "key=key" in renamed            # a local also used as a keyword name is left alone
    assert "except IndexError as err:" in renamed
    assert 'f"{len(out)}"' in renamed       # 'out' appears in an f-string, so it keeps its name
    assert "ordered" not in renamed and "j :=" not in renamed


def test_rename_locals_is_deterministic_per_rng():
    source = RENAME_SAMPLES["nested_loops"][0]
    a = mutation.rename_locals(source, random.Random(42))
    b = mutation.rename_locals(source, random.Random(42))
    assert a == b
    variants = {mutation.rename_locals(source, random.Random(s)) for s in range(8)}
    assert len(variants) > 1


def test_rename_locals_never_collides_with_existing_names():
    taken = " ".join(mutation._WORDS)
    source = f"def f(x):\n    y = x + 1\n    return y\n\n\nNAMES = '{taken}'\n" + "\n".join(
        f"{w} = 0" for w in mutation._WORDS) + "\n"
    renamed = mutation.rename_locals(source, random.Random(1))
    assert _exec_calls(renamed, "f", [[1], [5]]) == _exec_calls(source, "f", [[1], [5]])
    new = [n.id for n in ast.walk(ast.parse(renamed)) if isinstance(n, ast.Name) and n.id not in source]
    assert new and all(n not in mutation._WORDS for n in new)


def test_rename_locals_then_mutate_round_trip():
    renamed = mutation.rename_locals(SAMPLE, random.Random(11))
    for site in mutation.enumerate_sites(renamed):
        ast.parse(mutation.apply(renamed, [site]))
    assert len(mutation.enumerate_sites(renamed)) == len(mutation.enumerate_sites(SAMPLE))
