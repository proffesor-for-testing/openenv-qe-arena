"""Regression tests for defects found by independent review of the reward function.

Each test names the review that found the defect: Codex (correctness review of
ced30c9) or the red team (exploit hunt at 64304f5). All were reproduced before
they were fixed.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import solve_actions  # noqa: E402
from qe_arena import mutation, sandbox, tasks  # noqa: E402
from qe_arena.environment import QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, TASK_IDS, make_task  # noqa: E402


def play(task_id, seed, actions):
    env = QEEnvironment()
    env.reset(seed=seed, task_id=task_id)
    obs = None
    for action in actions:
        obs = env.step(QEAction(**action))
        if obs.done:
            break
    return obs


def _fault_sites(task):
    """Resolve the exact injected sites by reproducing the buggy file."""
    p = task.private
    candidates = [s for s in mutation.enumerate_sites(p["original"]) if s.description in p["sites"]]
    for combo in itertools.combinations(candidates, len(p["sites"])):
        try:
            if mutation.apply(p["original"], list(combo)) == p["buggy"]:
                return list(combo)
        except ValueError:
            continue
    raise AssertionError("could not reproduce the buggy file from its sites")


# Codex S1: qe_bugfix-d3 seed 37 paid 1.0 for fixing one of three faults, because the other two
# cancelled on the hidden inputs.
@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("seed", range(6))
def test_no_partial_fix_passes_the_hidden_tests(level, seed):
    task = make_task(f"qe_bugfix-d{level}", seed)
    p = task.private
    sites = _fault_sites(task)
    for size in range(1, len(sites)):
        for left_in in itertools.combinations(sites, size):  # fix everything except these
            partial = mutation.apply(p["original"], list(left_in))
            assert tasks.grade_bugfix(task, partial) == 0.0, [s.description for s in left_in]


# Red team #1: special-casing the bug report's input passed every hidden test in 12% of d1 tasks.
@pytest.mark.parametrize("seed", range(15))
def test_d1_special_casing_the_reported_input_earns_zero(seed):
    task = make_task("qe_bugfix-d1", seed)
    p = task.private
    buggy_out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    index = next(i for i, (r, o) in enumerate(zip(p["reference"], buggy_out)) if not sandbox.compare(r, o))
    ref = p["reference"][index]
    if ref.kind != "ok":
        pytest.skip("reported call raises; a value table cannot express it")
    renamed = p["buggy"].replace(f"def {p['entry']}(", "def impl_buggy(", 1)
    wrapper = (f"{renamed}\n\nREPORTED = {p['inputs'][index]!r}\nANSWER = {ref.value!r}\n\n"
               f"def {p['entry']}(*args):\n    if list(args) == REPORTED:\n        return ANSWER\n"
               f"    return impl_buggy(*args)\n")
    assert tasks.grade_bugfix(task, wrapper) == 0.0


@pytest.mark.parametrize("seed", range(10))
def test_d1_fault_is_revealed_by_several_distinct_calls(seed):
    task = make_task("qe_bugfix-d1", seed)
    p = task.private
    out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    revealing = {json.dumps(p["inputs"][i], sort_keys=True)
                 for i, (r, o) in enumerate(zip(p["reference"], out)) if not sandbox.compare(r, o)}
    assert len(revealing) >= tasks.MIN_REVEALING_D1


# Codex WHAT-ELSE 2 and red team #3: a 401-digit expected value crashed step() with OverflowError.
def test_huge_integer_expected_scores_without_crashing():
    obs = play("qe_mutant_kill-d1", 93, [{"op": "submit", "answer": {"tests": [
        {"args": ["login", "LOGIN"], "expected": 10**400}]}}])
    assert obs.done and 0.0 <= obs.reward <= 0.5


# Codex WHAT-ELSE 4 and red team #4: a lone surrogate in an edit crashed step() with UnicodeEncodeError.
def test_surrogate_in_an_edit_is_rejected_not_raised():
    env = QEEnvironment()
    env.reset(seed=0, task_id="qe_bugfix-d1")
    obs = env.step(QEAction(op="edit", path=MODULE, old="def ", new="#\ud800\ndef "))
    assert "UTF-8" in obs.error and not obs.done


# Red team #5: run_tests echoed a 393,466-character model-controlled value.
def test_run_tests_output_is_capped():
    task = make_task("qe_bugfix-d1", 2)
    huge = f"def {task.private['entry']}(*args):\n    return '9' * 300000\n"
    env = QEEnvironment()
    env.reset(seed=2, task_id=task.task_id)
    env.step(QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=huge))
    obs = env.step(QEAction(op="run_tests"))
    assert len(obs.content) < 2000


# Codex S5 and red team #2: a timeout counted as a kill, and a timeout during generation changed the task.
def test_a_mutant_timeout_is_never_a_kill(monkeypatch):
    task = make_task("qe_mutant_kill-d1", 93)
    p = task.private
    real = sandbox.run_calls

    def mutant_times_out(source, entry, calls, **kw):
        if source != p["original"]:
            return [sandbox.CallResult(kind="timeout", error="simulated load") for _ in calls]
        return real(source, entry, calls, **kw)

    env = QEEnvironment()
    env.reset(seed=93, task_id=task.task_id)  # generate under normal conditions first
    monkeypatch.setattr(sandbox, "run_calls", mutant_times_out)
    ref = real(p["original"], p["entry"], [p["inputs"][0]])[0]
    expected = {"raises": ref.value} if ref.kind == "exc" else ref.value
    obs = env.step(QEAction(op="submit", answer={"tests": [{"args": p["inputs"][0], "expected": expected}]}))
    assert obs.done and obs.reward == 0.0


@pytest.mark.parametrize("task_id", ["qe_bugfix-d1", "qe_mutant_kill-d1"])
def test_a_transient_timeout_does_not_change_the_generated_task(task_id, monkeypatch):
    calm = make_task(task_id, 0)
    real = sandbox.run_calls
    state = {"stalled": False}

    def stall_once(source, entry, calls, **kw):
        if not state["stalled"] and len(calls) > 1:
            state["stalled"] = True
            return [sandbox.CallResult(kind="timeout", error="simulated stall") for _ in calls]
        return real(source, entry, calls, **kw)

    monkeypatch.setattr(sandbox, "run_calls", stall_once)
    loaded = make_task(task_id, 0)
    assert state["stalled"]
    assert (loaded.prompt, loaded.files) == (calm.prompt, calm.files)


def test_a_transient_timeout_does_not_fail_a_correct_fix(monkeypatch):
    task = make_task("qe_bugfix-d1", 0)
    real = sandbox.run_calls
    state = {"stalled": False}

    def stall_once(source, entry, calls, **kw):
        if source == task.private["original"] and not state["stalled"]:
            state["stalled"] = True
            return [sandbox.CallResult(kind="timeout", error="simulated stall") for _ in calls]
        return real(source, entry, calls, **kw)

    monkeypatch.setattr(sandbox, "run_calls", stall_once)
    assert tasks.grade_bugfix(task, task.private["original"]) == 1.0
    assert state["stalled"]


# Codex S2: int-for-float leniency was top level only, and a returned {"raises": ...} dict was
# read as an exception expectation.
def test_expected_matching_follows_what_the_original_did():
    ok = lambda v: sandbox.CallResult(kind="ok", value=v)  # noqa: E731
    assert tasks._expected_matches({"p50": 7, "n": [1, 2]}, ok({"p50": 7.0, "n": [1.0, 2.0]}))
    assert not tasks._expected_matches(True, ok(1))
    assert not tasks._expected_matches(1, ok(True))
    assert tasks._expected_matches({"raises": "ValueError"}, ok({"raises": "ValueError"}))
    assert not tasks._expected_matches({"raises": "ValueError"}, ok(None))
    assert tasks._expected_matches({"raises": "ValueError"}, sandbox.CallResult(kind="exc", value="ValueError"))
    assert not tasks._expected_matches({"raises": "TypeError"}, sandbox.CallResult(kind="exc", value="ValueError"))
    assert not tasks._expected_matches(10**400, ok(1.0))
    assert not tasks._expected_matches(1, sandbox.CallResult(kind="timeout"))


# Codex S3: the old privacy test searched raw source inside JSON-escaped text, so it could not fail.
@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("seed", [0, 9])
def test_observation_text_never_contains_private_sources(task_id, seed):
    task = make_task(task_id, seed)
    env = QEEnvironment()
    reset = env.reset(seed=seed, task_id=task_id)
    shown = reset.prompt + env.step(QEAction(op="read", path="*")).content
    secrets = [task.private["original"]] if task.family == "qe_bugfix" else [m["source"] for m in task.private["mutants"]]
    for secret in secrets:
        body = secret.split("\n", 1)[1]  # past the shared def line
        assert body not in shown


# Codex S3 (coverage): the oracle was proven on only three seeds per cell.
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_oracle_earns_one_on_many_seeds(task_id):
    for seed in range(100, 108):
        assert play(task_id, seed, solve_actions(make_task(task_id, seed))).reward == 1.0, seed


# Codex WHAT-ELSE 1: glob_match's reference contradicted its docstring ("**" matches anything).
def test_glob_match_reference_follows_its_spec():
    from qe_arena import corpus

    assert corpus.call_reference("glob_match", ["**", "a\nb"]) is True
    assert corpus.call_reference("glob_match", ["*", "a\nb"]) is True
    assert corpus.call_reference("glob_match", ["*", "a/b"]) is False
