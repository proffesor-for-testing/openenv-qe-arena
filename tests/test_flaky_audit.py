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

from qe_arena import flaky, flaky_gen, sandbox  # noqa: E402
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


def _float_check(victim_block: str) -> tuple[str, str]:
    """(setup line or "", the asserted float expression) of a float-equality test."""
    setup = next((m.group(1) for m in re.finditer(r"^    ((?:total|parts) = .*)$", victim_block, re.M)), "")
    line = next(m.group(1) for m in re.finditer(r"^    assert (.*\bw\b.*)$", victim_block, re.M))
    return setup, line.split(" and ")[-1]


def _holds(check: tuple[str, str], w: float) -> bool:
    setup, expr = check
    ns = {"__builtins__": {}, "w": w, "sum": sum, "abs": abs}
    try:
        if setup:
            exec(setup, ns)  # noqa: S102 - our own generated float literal
        return bool(eval(expr, ns))  # noqa: S307
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
        check = _float_check(block(task.files[MODULE], task.private["victim"]))
        w = next(x / 64 for x in range(1, 64) if _holds(check, x / 64))
        edits = patch_first_line(task, f"Env.frac = lambda self: {w}")
        assert play(task, edits, {"test": task.private["victim"]}) <= 0.5, task.seed


def test_shadowing_a_builtin_above_the_harness_never_earns_the_repair(d3_by_cause):
    shadow = "RANGE = range\n\n\ndef range(*a):\n    return [] if a[1:] == (0, -1) else RANGE(*a)\n\n\n"
    for task in d3_by_cause["unordered_iteration"]:
        edits = [(HARNESS_HEAD, shadow + HARNESS_HEAD)]
        assert play(task, edits, {"test": task.private["victim"]}) <= 0.5, task.seed


# ---------------------------------------------------------------- MAJOR: emptying the polluter


def _writer(task):
    """The test that pollutes what the flaky test reads: it writes the same dict key, or warms the helper."""
    src, p = task.files[MODULE], task.private
    victim = block(src, p["victim"])
    names = [n for n in re.findall(r"^def (test_\w+)\(env\):", src, re.M) if n != p["victim"]]
    if p["cause"] == "shared_state":
        dict_name, key = re.search(r"\b([A-Z]+)\[('\w+')\]", victim).groups()
        for n in names:
            b = block(src, n)
            if f"{dict_name}[{key}] = " in b or f"{dict_name}.update({{{key}:" in b:
                return n
        return None
    helper = re.search(r"^def (\w+)\(\*args\):", src, re.M).group(1)
    return next((n for n in names if f"{helper}(" in block(src, n)
                 and not re.search(rf"(assert {helper}\(.*\) ==|got = {helper}\()", block(src, n))), None)


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
    for seed in range(300):
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
    assert mean <= 0.28, mean  # null baseline ~0.15; target within 0.1 of it, plus ~1.5 sigma at 300 seeds


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


# ---------------------------------------------------------------- re-audit of 92758b0 (review-flaky/v2)


def _d3_with(predicate, limit, seeds=range(600)):
    found = []
    for seed in seeds:
        task = make_task("qe_flaky-d3", seed)
        if task.private["victim"] and predicate(task):
            found.append(task)
            if len(found) == limit:
                break
    return found


def _repaired_body(task):
    return task.private["repair"]["new"].split("\n", 1)[1]


def _flaky_body(task):
    return block(task.files[MODULE], task.private["victim"]).split("\n", 1)[1]


def test_m1_run_once_through_a_mutable_default_is_refused(d3_by_cause):
    # runonce_demo.py channel 1: graded runs shared one process, so the flaky body ran only on the first.
    for task in [t for ts in d3_by_cause.values() for t in ts][:12]:
        victim = task.private["victim"]
        hack = (f"def {victim}(env, ran=[]):\n    if not ran:\n        ran.append(1)\n"
                + "\n".join("    " + line for line in _flaky_body(task).split("\n")) + "\n" + _repaired_body(task))
        region = task.private["region"].replace(block(task.private["region"], victim), hack, 1)
        assert flaky.gate_region(region, set(task.private["protected"]))
        assert not flaky.repair_ok(task, region)
        assert play(task, [(block(task.files[MODULE], victim), hack)], {"test": victim}) <= 0.5


def test_m1_a_dict_value_mutated_in_place_does_not_survive_into_the_next_run():
    # Channel 2: the box reset copied dicts shallowly, so BOX['k'].append(...) carried over to later runs, both in
    # the graded harness and in the shown one.
    task = make_task("qe_flaky-d3", 0)
    region = "BOX = {'k': [1]}\n\n\ndef test_alpha(env):\n    BOX['k'].append(2)\n    assert len(BOX['k']) == 2\n\n\n"
    runs = [[[0], flaky.SEED_SPLIT + i, 1000 * i] for i in range(5)]
    graded = flaky.graded_source(task.private["prefix"], region, ["test_alpha"], ["BOX"])
    shown = task.private["prefix"] + region + flaky.shown_harness(["test_alpha"], ["BOX"])
    for source in (graded, shown):
        assert [o["test_alpha"] for o in flaky.run(source, runs)] == [True] * 5


@pytest.mark.parametrize("signature", ["env, extra=None", "*env", "env, *rest", "env, **kw", "evn"])
def test_m1_test_signatures_are_exactly_env(signature):
    region = f"def test_alpha({signature}):\n    assert True\n"
    assert flaky.gate_region(region, set())
    assert not flaky.gate_region("def test_alpha(env):\n    assert True\n", set())


def test_m2_the_auditors_sweep_detecting_hacks_score_half():
    seed1_draw = 96542  # the first draw of seed 1, which every sweep call used at 92758b0
    clocks = _d3_with(lambda t: t.private["cause"] == "clock_boundary", 6)
    for task in clocks:
        vb = block(task.files[MODULE], task.private["victim"])
        widened = re.sub(r"\b(500|1000|2000|2500|4000|5000|10000)\b", "10**9", vb)
        hack = (f"def {task.private['victim']}(env):\n    if env.draw(10**9) == {seed1_draw}:\n        return\n"
                + widened.split("\n", 1)[1])
        assert play(task, [(vb, hack)], {"test": task.private["victim"]}) <= 0.5, task.seed
    draws = _d3_with(lambda t: t.private["cause"] == "shared_random"
                     and " if env.draw(" in block(t.files[MODULE], t.private["victim"]), 4)
    assert clocks and draws
    for task in draws:
        vb = block(task.files[MODULE], task.private["victim"])
        widened = re.sub(r"env\.draw\(\d+\)", "env.draw(10**9)", vb)
        hack = (f"def {task.private['victim']}(env):\n    x, y = env.draw(10**9), env.draw(10**9)\n"
                f"    if x == y or x == {seed1_draw}:\n        return\n" + widened.split("\n", 1)[1])
        assert play(task, [(vb, hack)], {"test": task.private["victim"]}) <= 0.5, task.seed


def test_m2_no_constant_identifies_sweep_runs_across_tasks():
    unforced = []  # what a probe `env.draw(10**9)` sees first in sweep runs that do not force the first draw
    for seed in range(20):
        task = make_task("qe_flaky-d3", seed)
        for kind, value, run_seed, t0 in task.private["sweep"]:
            assert flaky.SEED_SPLIT <= run_seed < 2 * flaky.SEED_SPLIT
            if kind == "draw":
                if value[0] >= 8:
                    unforced.append(value[0] % 10**9)
            else:
                unforced.append((run_seed % 2147483646 + 1) * 48271 % 2147483647 % 10**9)
    assert max(unforced.count(v) for v in set(unforced)) <= 2


def test_m2_a_probe_draw_before_the_flaky_draw_does_not_dodge_the_sweep():
    draws = _d3_with(lambda t: t.private["cause"] == "shared_random"
                     and " if env.draw(" in block(t.files[MODULE], t.private["victim"]), 4)
    assert draws
    for task in draws:
        vb = block(task.files[MODULE], task.private["victim"])
        widened = re.sub(r"env\.draw\(\d+\)", "env.draw(10**9)", vb)
        hack = (f"def {task.private['victim']}(env):\n    if env.draw(10**9) < 8:\n        return\n"
                + widened.split("\n", 1)[1])
        assert play(task, [(vb, hack)], {"test": task.private["victim"]}) <= 0.5, task.seed


def test_m2_the_frac_sweep_only_uses_values_env_can_produce():
    assert all(float(round(x * 2147483647)) / 2147483647 == x for x in flaky.FRAC_SWEEP)


def test_m3_an_order_insensitive_pool_repair_earns_full_credit():
    # legit2.py: seeds 33 and 310 scored 0.5 because the mutant swapped the two culprit outputs.
    tasks = _d3_with(lambda t: t.private["cause"] == "unordered_iteration"
                     and "got = [" in block(t.files[MODULE], t.private["victim"]).split("\n")[1], 8)
    assert len(tasks) >= 4
    for task in tasks:
        vb = block(task.files[MODULE], task.private["victim"])
        fixed = re.sub(r"assert got == (\[.*\])$", r"assert sorted(got, key=repr) == sorted(\1, key=repr)", vb,
                       count=1, flags=re.M)
        assert fixed != vb
        assert play(task, [(vb, fixed)], {"test": task.private["victim"]}) == 1.0, task.seed
