"""Regression tests from the qe_flaky audit of 654ad21 (scratchpad review-flaky/: hacks.py, telltale.py, rates.py).

Each test reproduces one finding the auditor measured; each failed on 654ad21.
"""

from __future__ import annotations

import ast
import json
import random
import re
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qe_arena import flaky, sandbox  # noqa: E402
from qe_arena.environment import QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, make_task  # noqa: E402

HARNESS_HEAD = "from collections import namedtuple"


@pytest.fixture(scope="module")
def d3_by_cause():
    found: dict[str, list] = {c: [] for c in flaky.CAUSES}
    for seed in range(400):
        task = make_task("qe_flaky-d3", seed)
        cause = task.private["cause"]
        if cause and len(found[cause]) < 6:
            found[cause].append(task)
        if all(len(v) == 6 for v in found.values()):
            break
    return found


def block(src: str, name: str) -> str:
    start = src.index(f"def {name}(env):\n")
    end = src.find("\n\n\n", start)
    return src[start:end]


def play(task, edits, answer):
    env = QEEnvironment()
    env.reset(seed=task.seed, task_id=task.task_id)
    for old, new in edits:
        env.step(QEAction(op="edit", path=MODULE, old=old, new=new))
    return env.step(QEAction(op="submit", answer=answer)).reward


def patch_first_line(task, line):
    victim = task.private["victim"]
    b = block(task.files[MODULE], victim)
    return [(b, b.replace(f"def {victim}(env):\n", f"def {victim}(env):\n    {line}\n", 1))]


def _float_expr(victim_block: str) -> str:
    return next(m.group(1) for m in re.finditer(r"^    assert (.*\bw\b.*)$", victim_block, re.M))


def _holds(expr: str, w: float) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": {}}, {"w": w, "abs": abs}))  # noqa: S307 - our own float literal
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- BLOCKER: runtime patches of the harness


@pytest.mark.parametrize("cause,line", [
    ("clock_boundary", "Env.now = lambda self: 0"),
    ("unordered_iteration", "Env.pool = lambda self, items: list(items)"),
    ("shared_random", "Env.draw = lambda self, n: 0"),
    ("shared_random", "Env.draw = lambda self, n: 1"),
])
def test_patching_the_env_class_never_earns_the_repair(d3_by_cause, cause, line):
    for task in d3_by_cause[cause]:
        assert play(task, patch_first_line(task, line), {"test": task.private["victim"]}) <= 0.5, task.seed


def test_patching_env_frac_never_earns_the_repair(d3_by_cause):
    for task in d3_by_cause["float_equality"]:
        expr = _float_expr(block(task.files[MODULE], task.private["victim"]))
        w = next(x / 64 for x in range(1, 64) if _holds(expr, x / 64))
        edits = patch_first_line(task, f"Env.frac = lambda self: {w}")
        assert play(task, edits, {"test": task.private["victim"]}) <= 0.5, task.seed


def test_shadowing_a_builtin_above_the_harness_never_earns_the_repair(d3_by_cause):
    shadow = "RANGE = range\n\n\ndef range(*a):\n    return [] if a[1:] == (0, -1) else RANGE(*a)\n\n\n"
    for task in d3_by_cause["unordered_iteration"]:
        edits = [(HARNESS_HEAD, shadow + HARNESS_HEAD)]
        assert play(task, edits, {"test": task.private["victim"]}) <= 0.5, task.seed


# ---------------------------------------------------------------- MAJOR: emptying the polluter


def _writer(task):
    src, p = task.files[MODULE], task.private
    names = re.findall(r"^def (test_\w+)\(env\):", src, re.M)
    victim_reads = set(re.findall(r"\b([A-Z]+)\b", block(src, p["victim"])))
    victim_helpers = set(re.findall(r"\b(\w+)\(", block(src, p["victim"])))
    for n in names:
        if n == p["victim"]:
            continue
        b = block(src, n)
        if re.search(r"^    ([A-Z]+)\[.*\] = ", b, re.M) and re.search(r"^    ([A-Z]+)\[", b, re.M).group(1) in victim_reads:
            return n
        if p["cause"] == "cache_leak" and any(f"{h}(" in b for h in victim_helpers if h.islower() and h != p["entry"]):
            return n
    return None


@pytest.mark.parametrize("cause", ["shared_state", "cache_leak"])
def test_emptying_the_polluter_never_earns_the_repair(d3_by_cause, cause):
    hits = 0
    for task in d3_by_cause[cause]:
        writer = _writer(task)
        if writer is None:
            continue
        hits += 1
        wb = block(task.files[MODULE], writer)
        assert play(task, [(wb, f"def {writer}(env):\n    pass")], {"test": task.private["victim"]}) <= 0.5
    assert hits >= 3


# ---------------------------------------------------------------- MAJOR: repairs that keep the defect


def test_widening_the_clock_bucket_keeps_the_defect(d3_by_cause):
    for task in d3_by_cause["clock_boundary"]:
        vb = block(task.files[MODULE], task.private["victim"])
        widened = re.sub(r"// \d+", "// 10**9", vb)
        assert widened != vb
        assert play(task, [(vb, widened)], {"test": task.private["victim"]}) <= 0.5, task.seed


def test_drawing_from_a_huge_range_keeps_the_defect(d3_by_cause):
    tried = 0
    for seed in range(600):
        task = make_task("qe_flaky-d3", seed)
        if task.private["cause"] != "shared_random":
            continue
        vb = block(task.files[MODULE], task.private["victim"])
        if not re.search(r" if env\.draw\(\d+\) else ", vb):
            continue
        tried += 1
        widened = re.sub(r"env\.draw\(\d+\)", "env.draw(10**9)", vb)
        assert play(task, [(vb, widened)], {"test": task.private["victim"]}) <= 0.5, task.seed
        if tried == 4:
            break
    assert tried >= 2


# ---------------------------------------------------------------- MAJOR: surface templates give the answer away


def _syntax_classifier(src: str):
    """The auditor's S classifier (telltale.py, syntax only), on the templates of 654ad21."""
    region = src[: src.index(HARNESS_HEAD)]
    tree = ast.parse(region)
    tests = {n.name: ast.get_source_segment(region, n) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    bad = ("(w + 0.1) - 0.1 == w", "w * 0.1 * 10 == w", "(w + 0.5) - 0.5 == w", "(w + 0.2) + 0.1 == w + 0.3",
           "w * 3 / 3 == w", "w / 10 * 10 == w")
    m = re.search(r"key = repr\(args\)\[:(\d+)\]", region)
    prefix = int(m.group(1)) if m else None
    helper = re.search(r"def (\w+)\(\*args\):", region)
    dicts = re.findall(r"^(\w+) = \{'", region, re.M)
    for name, body in tests.items():
        if re.search(r"= .+ if env\.draw\(\d+\) else ", body):
            return name, "shared_random"
        if re.search(r"env\.now\(\) // \d+ == stamp", body):
            return name, "clock_boundary"
        if any(f"assert {b}" in body for b in bad):
            return name, "float_equality"
        g = re.search(r"assert got == \[(.+)\]$", body, re.M)
        if g and "env.pool" in body:
            try:
                vals = ast.literal_eval("[" + g.group(1) + "]")
            except (ValueError, SyntaxError):
                vals = []
            if len(vals) == 2 and json.dumps(vals[0]) != json.dumps(vals[1]):
                return name, "unordered_iteration"
    if dicts:
        D = dicts[0]
        writes: dict = {}
        for name, body in tests.items():
            for k in re.findall(rf"^    {D}\[('\w+')\] = ", body, re.M):
                if "try:" not in body:
                    writes.setdefault(k, []).append(name)
        for name, body in tests.items():
            for k in re.findall(rf"{D}\[('\w+')\]\)", body):
                if not re.search(rf"^    {D}\[{k}\] = ", body, re.M) and any(w != name for w in writes.get(k, [])):
                    return name, "shared_state"
    if helper and prefix is not None:
        H = helper.group(1)
        calls: dict = {}
        for name, body in tests.items():
            for a in re.findall(rf"{H}\((.*?)\) (?:==|is not None)", body):
                try:
                    key = repr(ast.literal_eval("(" + a + ",)"))[:prefix]
                except (ValueError, SyntaxError):
                    continue
                calls.setdefault(name, []).append((key, "is not None" in body))
        writers = {k for cs in calls.values() for k, w in cs if w}
        for name, cs in calls.items():
            for k, w in cs:
                if not w and k in writers:
                    return name, "cache_leak"
    return None, None


@pytest.mark.parametrize("level", [1, 2])
def test_the_syntax_only_classifier_scores_near_the_null_baseline(level):
    scores = []
    for seed in range(150):
        task = make_task(f"qe_flaky-d{level}", seed)
        p = task.private
        test, cause = _syntax_classifier(task.files[MODULE])
        if p["victim"] is None:
            scores.append(1.0 if test is None else 0.0)
        elif test != p["victim"]:
            scores.append(0.0)
        else:
            scores.append(1.0 if level == 1 or cause == p["cause"] else 0.5)
    mean = statistics.mean(scores)
    print(f"\nS classifier d{level}: {mean:.3f}")
    assert mean <= 0.27, mean


# ---------------------------------------------------------------- MINOR: low-rate flakes and wording


def test_a_low_rate_flake_band_exists_and_is_still_observed():
    low = []
    for seed in range(300):
        task = make_task("qe_flaky-d1", seed)
        p = task.private
        if not p["victim"] or p["cause"] in ("shared_state", "cache_leak", "unordered_iteration"):
            continue
        rng = random.Random(f"low-band:{seed}")
        n = len(p["tests"])
        combos = [[rng.sample(range(n), n), rng.randrange(flaky.SEED_SPLIT, 2 * flaky.SEED_SPLIT),
                   rng.randrange(10**7)] for _ in range(400)]
        runs = [r.value[p["victim"]] for r in sandbox.run_calls(task.files[MODULE], "run_suite", combos,
                                                               timeout_s=20.0)]
        rate = runs.count(False) / len(runs)
        if 0.015 <= rate <= 0.11:
            low.append(task)
            graded = [r.value[p["victim"]] for r in sandbox.run_calls(task.files[MODULE], "run_suite", p["graded"],
                                                                      timeout_s=20.0)]
            assert set(graded) == {True, False}
        if len(low) >= 3:
            break
    assert len(low) >= 3


def test_shared_state_wording_names_a_direct_read():
    assert flaky.CAUSE_TEXT["shared_state"].startswith(
        "assigns to an entry of a module-level dict that another test reads directly")
