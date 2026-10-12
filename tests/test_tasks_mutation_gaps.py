"""Task generation and grading under the WASM sandbox: settled outcomes, retries, cancellation check,
pinned generation constants. Added after mutation testing b43f503 (30 of 43 tasks.py survivors)."""

from __future__ import annotations

import itertools
import json

import pytest

from qe_arena import mutation, sandbox, tasks
from qe_arena.tasks import make_task

CR = sandbox.CallResult


def test_generation_constants_are_pinned_literals():
    # Kills FAULTS / MIN_SUBTLE / MIN_REVEALING_D1 / MAX_SHOWN / HIDDEN_INPUTS off-by-ones that the generator
    # contract tests read back symbolically (tests compare with tasks.MIN_REVEALING_D1 itself).
    assert tasks.FAULTS == {1: 1, 2: 2, 3: 3} and tasks.MIN_SUBTLE == {1: 0, 2: 1, 3: 2}
    assert tasks.MIN_REVEALING_D1 == 3 and tasks.MAX_SHOWN == 200 and tasks.HIDDEN_INPUTS == 48
    assert tasks.RESOURCE_EXCEPTIONS == ("MemoryError", "RecursionError")


def test_clip_boundary():
    # Kills E-tasks-74 (<= -> <, slice shifts) and E-tasks-70 via behaviour, M-clip-none, M-result-text-exc-unclipped.
    assert tasks._clip("x" * 200) == "x" * 200
    assert tasks._clip("x" * 201) == "x" * 200 + "... (201 chars)"
    assert tasks._result_text(CR("exc", "E" * 300)) == "raises " + "E" * 200 + "... (300 chars)"
    assert tasks._result_text(CR("ok", "y" * 300)).endswith("... (302 chars)")


@pytest.mark.parametrize("result, settled", [
    (CR("ok", None), True), (CR("exc", "ValueError"), True), (CR("exc", "KeyError"), True),
    (CR("exc", "MemoryError"), False), (CR("exc", "RecursionError"), False),
    (CR("fuel"), False), (CR("resource"), False), (CR("crash"), False), (CR("skipped"), False), (CR("infra"), False),
    (CR("timeout"), False), (CR("rejected"), False),
])
def test_settled_matrix(result, settled):
    # Kills the _settled engine/manual survivors (fuel/resource counted as settled, resource names dropped).
    assert tasks._settled(result) is settled
    assert tasks._usable([CR("ok", 1), result]) is settled
    assert (tasks._killing([CR("ok", 1)], [result]) == [0]) is (settled and not sandbox.compare(CR("ok", 1), result))
    assert (tasks._killing([result], [CR("ok", 2)]) == [0]) is (settled and not sandbox.compare(result, CR("ok", 2)))


def _counting_run_calls(monkeypatch, first_kinds):
    seen = []

    def fake(source, entry, calls, **kw):
        seen.append(len(calls))
        if len(seen) == 1:
            return [CR(k) for k in first_kinds]
        return [CR("ok", 0) for _ in calls]
    monkeypatch.setattr(tasks.sandbox, "run_calls", fake)
    return seen


@pytest.mark.parametrize("first, retried", [
    (["ok", "infra"], True), (["timeout", "ok"], True), (["infra", "infra"], True),
    (["fuel", "ok"], False), (["resource", "crash"], False),
])
def test_outputs_retries_only_transient_infra(monkeypatch, first, retried):
    # Kills M-outputs-no-retry / -retry-all / -retry-infra-only: one infra row retries the batch once,
    # deterministic limits never do.
    first = [k if k != "ok" else "ok" for k in first]
    seen = _counting_run_calls(monkeypatch, first)
    tasks._outputs("def f(x):\n    return 0\n", "f", [[1], [2]])
    assert len(seen) == (2 if retried else 1)


def test_subsets_must_each_fail_settled_on_some_input(monkeypatch):
    # Kills M-subsets-skip-largest / -singletons-only / -drop-usable and the _subsets_all_fail engine survivors.
    original = "def f(x):\n    a = x + 1\n    b = a - 1\n    return b * 2\n"
    sites = {s.description: s for s in mutation.enumerate_sites(original)}
    plus = next(s for d, s in sites.items() if "'+' -> '-'" in d)
    minus = next(s for d, s in sites.items() if "'-' -> '+'" in d)
    times = next(s for d, s in sites.items() if "'*' -> '//'" in d)
    inputs = [[0], [3], [8]]
    reference = sandbox.run_calls(original, "f", inputs)
    assert tasks._subsets_all_fail(original, [plus, times], "f", inputs, reference)
    # plus and minus cancel: fixing only `times` leaves them in and passes every hidden input
    assert not tasks._subsets_all_fail(original, [plus, minus, times], "f", inputs, reference)
    real = tasks._outputs
    monkeypatch.setattr(tasks, "_outputs", lambda src, e, calls: [CR("fuel")] + real(src, e, calls)[1:])
    assert not tasks._subsets_all_fail(original, [plus, times], "f", inputs, reference)


def test_pool_is_validated_and_restricts_the_corpus():
    # Kills M-make-pool-unchecked / M-make-pool-ignored and the make_task pool engine survivors.
    for bad in ((), ("no_such_function",)):
        with pytest.raises(ValueError):
            make_task("qe_bugfix-d1", 0, pool=bad)
    name = make_task("qe_bugfix-d1", 0).private["corpus"]
    other = next(n for n in sorted(tasks.corpus.ENTRIES) if n != name)
    for task_id in ("qe_bugfix-d1", "qe_mutant_kill-d1"):
        assert make_task(task_id, 0, pool=(other,)).private["corpus"] == other


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_subtle_quota_and_fault_count_per_level(level, seed):
    # Kills M-faults-*, M-minsubtle-zero, M-bf-drop-reserve, M-bf-reserve-gt with literal levels.
    task = make_task(f"qe_bugfix-d{level}", seed)
    p = task.private
    assert len(p["sites"]) == level
    pool = dict.fromkeys([*mutation.enumerate_sites(p["original"]), *p.get("fault_sites", [])])  # real faults too
    cands = [s for s in pool if s.description in p["sites"]]
    sites = next(list(c) for c in itertools.combinations(cands, level)
                 if _apply_or_none(p["original"], list(c)) == p["buggy"])
    subtle = 0
    for s in sites:
        out = sandbox.run_calls(mutation.apply(p["original"], [s]), p["entry"], p["inputs"])
        kills = tasks._killing(p["reference"], out)
        assert kills and tasks._usable(out)
        subtle += len(kills) <= 0.15 * 48
    assert subtle >= {2: 1, 3: 2}[level]


def _apply_or_none(src, sites):
    try:
        return mutation.apply(src, sites)
    except ValueError:
        return None


@pytest.mark.parametrize("seed", [*range(12), 19, 20])  # without the filter, seed 20 draws a fault one hidden call reveals
def test_d1_fault_revealed_by_at_least_three_distinct_hidden_calls_literal(seed):
    # Kills E-tasks-28 (MIN_REVEALING_D1 3 -> 2) and M-bf-min-revealing-count-raw: the existing test compares
    # with tasks.MIN_REVEALING_D1 itself.
    p = make_task("qe_bugfix-d1", seed).private
    out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    revealing = {json.dumps(p["inputs"][i], sort_keys=True) for i in tasks._killing(p["reference"], out)}
    assert len(revealing) >= 3


def test_expected_matching_coerces_nested_ints_but_never_bools():
    # Kills M-aat-no-list / -no-dict / -isinstance / -any-to-float and _as_actual_types engine survivors.
    ok = lambda v: CR("ok", v)  # noqa: E731
    assert tasks._expected_matches([1, [2]], ok([1.0, [2.0]]))
    assert tasks._expected_matches({"a": {"b": 3}}, ok({"a": {"b": 3.0}}))
    assert not tasks._expected_matches([True], ok([1.0]))
    assert not tasks._expected_matches("1", ok(1.0))
    assert not tasks._expected_matches([1, 2], ok([1.0]))
    assert not tasks._expected_matches({"a": 1}, ok({"b": 1.0}))
    assert tasks._expected_matches({"a": 1, "x": 2}, ok({"a": 1.0, "x": 2})) is True



def test_a_subset_that_is_equivalent_alone_fails_the_check():
    # Kills E-tasks-124 (1 -> 2: singletons skipped) as a unit, independent of the generator's own filter.
    original = "def f(x):\n    if x > 100:\n        return 1\n    return x * 2\n"
    sites = mutation.enumerate_sites(original)
    ge = next(s for s in sites if "'>' -> '>='" in s.description)
    times = next(s for s in sites if "'*' -> '//'" in s.description)
    inputs = [[0], [3], [8]]
    reference = sandbox.run_calls(original, "f", inputs)
    assert not tasks._subsets_all_fail(original, [ge, times], "f", inputs, reference)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", range(8))
def test_buggy_file_settles_on_every_hidden_and_shown_input(level, seed):
    # Kills E-tasks-214/225 and M-bf-drop-combined-usable / -shown-usable: a buggy file that loops or
    # crashes on a graded or shown input makes fuel, not behaviour, decide the task.
    p = make_task(f"qe_bugfix-d{level}", seed).private
    assert tasks._usable(sandbox.run_calls(p["buggy"], p["entry"], p["inputs"]))
    shown = [v["args"] for v in p["visible"]] + ([p["report_args"]] if p["report_args"] is not None else [])
    assert tasks._usable(sandbox.run_calls(p["buggy"], p["entry"], shown))
    assert all(tasks._settled(v["expected"]) for v in p["visible"])


def test_multi_fault_generation_consults_the_cancellation_check(monkeypatch):
    # Kills M-subsets-call-dropped (and call-site and/or swaps). Cancelling faults are rare on the current
    # corpus (0 of 120 bugfix tasks for seeds 0-39 change without the check), so a seeded contract test
    # cannot see it; vetoing every combination must make d2 ungeneratable.
    name = make_task("qe_bugfix-d2", 0).private["corpus"]
    seen = []

    def veto(original, sites, entry, inputs, reference):
        seen.append(len(sites))
        return False
    monkeypatch.setattr(tasks, "_subsets_all_fail", veto)
    with pytest.raises(RuntimeError):
        make_task("qe_bugfix-d2", 0, pool=(name,))
    assert seen and set(seen) == {2}


def test_visible_run_retries_a_transient_infra_failure(monkeypatch):
    # Kills M-run-visible-run-calls: one worker hiccup must not show the model FAILs on passing code.
    task = make_task("qe_bugfix-d1", 9)
    real = sandbox.run_calls
    state = {"n": 0}

    def flaky(source, entry, calls, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return [CR("infra", error="simulated") for _ in calls]
        return real(source, entry, calls, **kw)
    monkeypatch.setattr(tasks.sandbox, "run_calls", flaky)
    report = tasks.run_visible(task, task.private["original"])
    assert report.startswith(f"{len(task.private['visible'])}/{len(task.private['visible'])} visible tests passed")


def test_long_visible_values_are_clipped_in_run_tests():
    # The existing run_tests cap test returns 300,000 characters, which the sandbox rejects as too large
    # (a crash), so it never reaches _clip. 100,000 characters is under the value cap and must be clipped.
    task = make_task("qe_bugfix-d1", 2)
    huge = f"def {task.private['entry']}(*args):\n    return '9' * 100000\n"
    report = tasks.run_visible(task, huge)
    assert len(report) < 2000 and "(100002 chars)" in report


def test_expected_matching_never_raises_on_mismatched_container_types():
    # Kills E-tasks-349 (and -> or): a list or string expected against a dict result raised AttributeError,
    # which _expected_matches does not catch ("never raises" in its docstring).
    ok = lambda v: CR("ok", v)  # noqa: E731
    assert tasks._expected_matches([1], ok({"a": 1.0})) is False
    assert tasks._expected_matches("a", ok({"a": 1.0})) is False
    assert tasks._expected_matches({"a": 1}, ok([1.0])) is False


# ---------------------------------------------------------------- added on the merged head (team lead HIGH b)

@pytest.mark.parametrize("mutant", [
    "def f(x):\n    while True:\n        pass\n",                    # fuel
    "def f(x):\n    return len([0] * 10 ** 9)\n",                     # resource (memory)
    "def f(x):\n    return f(x)\n",                                  # resource (recursion)
    "def f(x):\n    s = []\n    for _ in range(100000):\n        s = [s]\n    return str(s)\n",  # wasm stack
])
def test_a_fuel_or_resource_mutant_outcome_is_never_a_kill(mutant):
    # HIGH (b): the existing test only simulated "timeout". Real fuel and resource outcomes from the WASI
    # sandbox must earn nothing, for an exact expected value or not.
    original = "def f(x):\n    return x + 1\n"
    task = tasks.Task(task_id="qe_mutant_kill-d1", family="qe_mutant_kill", difficulty=1, seed=0, prompt="",
                      files={}, private={"original": original, "entry": "f",
                                         "mutants": [{"source": mutant, "site": "", "random_kill_rate": 0.5}]})
    [out] = sandbox.run_calls(mutant, "f", [[3]], fuel=500_000_000)
    assert out.kind in ("fuel", "resource")
    assert tasks._killing([CR("ok", 4)], [out]) == []
    assert tasks.grade_mutant_kill(task, {"tests": [{"args": [3], "expected": 4}]}) == 0.0
