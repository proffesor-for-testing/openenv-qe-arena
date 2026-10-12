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
    pool = dict.fromkeys([*mutation.enumerate_sites(p["original"]), *p["fault_sites"]])  # real faults too
    candidates = [s for s in pool if s.description in p["sites"]]
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
    reported = p["report_args"]
    [ref] = sandbox.run_calls(p["original"], p["entry"], [reported])
    assert ref.kind in ("ok", "exc")
    # Exceptions compare by type name, so a raising reference is special-cased by raising the same type.
    answer = f"return {ref.value!r}" if ref.kind == "ok" else f"raise {ref.value}('special-cased')"
    renamed = p["buggy"].replace(f"def {p['entry']}(", "def impl_buggy(", 1)
    wrapper = (f"{renamed}\n\nREPORTED = {reported!r}\n\n"
               f"def {p['entry']}(*args):\n    if list(args) == REPORTED:\n        {answer}\n"
               f"    return impl_buggy(*args)\n")
    [special] = sandbox.run_calls(wrapper, p["entry"], [reported])
    assert sandbox.compare(ref, special)  # the wrapper really answers the reported call correctly
    assert tasks.grade_bugfix(task, wrapper) == 0.0


# Red team recheck N02: visible tests showed correct outputs for graded edge cases, so a table learned
# across episodes could pass tasks whose faults show only there.
@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", range(8))
def test_nothing_shown_is_ever_graded(level, seed):
    task = make_task(f"qe_bugfix-d{level}", seed)
    p = task.private
    graded = {json.dumps(a, sort_keys=True) for a in p["inputs"]}
    shown = [v["args"] for v in p["visible"]] + ([p["report_args"]] if p["report_args"] is not None else [])
    assert shown and not graded.intersection(json.dumps(a, sort_keys=True) for a in shown)


# Red team recheck N04: a MemoryError from the sandbox's 256 MiB cap counted as a kill on Linux.
def test_resource_exceptions_are_never_a_behavioural_difference():
    ok = sandbox.CallResult(kind="ok", value=None)
    for name in ("MemoryError", "RecursionError"):
        exc = sandbox.CallResult(kind="exc", value=name)
        assert tasks._killing([ok], [exc]) == [] and tasks._killing([exc], [ok]) == []
        assert not tasks._usable([exc])
    assert tasks._killing([ok], [sandbox.CallResult(kind="exc", value="ValueError")]) == [0]


@pytest.mark.parametrize("seed", range(10))
def test_d1_fault_is_revealed_by_several_distinct_calls(seed):
    task = make_task("qe_bugfix-d1", seed)
    p = task.private
    out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    revealing = {json.dumps(p["inputs"][i], sort_keys=True)
                 for i, (r, o) in enumerate(zip(p["reference"], out)) if not sandbox.compare(r, o)}
    assert len(revealing) >= 3  # literal: comparing with tasks.MIN_REVEALING_D1 let a 3 -> 2 mutant survive


# Codex WHAT-ELSE 2 and red team #3: a 401-digit expected value crashed step() with OverflowError.
def test_huge_integer_expected_scores_without_crashing():
    obs = play("qe_mutant_kill-d1", 93, [{"op": "submit", "answer": {"tests": [
        {"args": ["login", "LOGIN"], "expected": 10**400}]}}])
    assert obs.done and 0.0 <= obs.reward <= 0.5


# Codex WHAT-ELSE 4 and red team #4: a lone surrogate in an edit crashed step() with UnicodeEncodeError.
def test_surrogate_in_an_edit_is_rejected_not_raised():
    env = QEEnvironment()
    env.reset(seed=0, task_id="qe_bugfix-d1")
    entry = make_task("qe_bugfix-d1", 0).private["entry"]  # a module may also define helpers
    obs = env.step(QEAction(op="edit", path=MODULE, old=f"def {entry}(", new=f"#\ud800\ndef {entry}("))
    assert "UTF-8" in obs.error and not obs.done


# Red team #5: run_tests echoed a 393,466-character model-controlled value.
def test_run_tests_output_is_capped():
    task = make_task("qe_bugfix-d1", 2)
    # 100,000 characters: under the sandbox's 128 KiB value cap (300,000 became a crash and never reached _clip)
    huge = f"def {task.private['entry']}(*args):\n    return '9' * 100000\n"
    env = QEEnvironment()
    env.reset(seed=2, task_id=task.task_id)
    env.step(QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=huge))
    obs = env.step(QEAction(op="run_tests"))
    assert len(obs.content) < 2000 and "(100002 chars)" in obs.content


# Codex S5 and red team #2: a timeout counted as a kill, and a timeout during generation changed the task.
# Since the WASI review (P5) a persistent infrastructure failure is not scored at all: it is an error,
# never a kill and never a 0, and the same submission is graded normally once the sandbox recovers.
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
    ref = real(p["original"], p["entry"], [p["inputs"][0]])[0]
    expected = {"raises": ref.value} if ref.kind == "exc" else ref.value
    answer = {"tests": [{"args": p["inputs"][0], "expected": expected}]}
    monkeypatch.setattr(sandbox, "run_calls", mutant_times_out)
    with pytest.raises(sandbox.InfraError):
        env.step(QEAction(op="submit", answer=answer))
    assert not env._done and env._reward == 0.0 and env.state.step_count == 0
    monkeypatch.setattr(sandbox, "run_calls", real)
    obs = env.step(QEAction(op="submit", answer=answer))
    assert obs.done and obs.reward == tasks.grade_mutant_kill(task, answer)


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


# Codex review of the WASI sandbox, P3: a trap ends its batch and skips the calls after it, so mutant
# kill tests must not share a batch, or a test that loops the mutant would cancel a later killing test.
def test_mutant_kill_credit_does_not_depend_on_test_order():
    original = "def f(x):\n    return x + 1\n"
    mutant = "def f(x):\n    while x == 0:\n        pass\n    return x + 2\n"
    task = tasks.Task(task_id="qe_mutant_kill-d3", family="qe_mutant_kill", difficulty=3, seed=0, prompt="",
                      files={}, private={"original": original, "entry": "f",
                                         "mutants": [{"source": mutant, "site": "", "random_kill_rate": 0.5}]})
    looping, killing = {"args": [0], "expected": 1}, {"args": [5], "expected": 6}
    assert tasks.grade_mutant_kill(task, {"tests": [killing, looping]}) == 1.0
    assert tasks.grade_mutant_kill(task, {"tests": [looping, killing]}) == 1.0
    assert tasks.grade_mutant_kill(task, {"tests": [looping]}) == 0.0  # a loop is never a kill


# Codex review of the WASI sandbox, P5: a persistent infrastructure failure graded as 0, so a correct
# submission could earn different rewards depending on the machine. It must be an error frame instead.
def _always_infra(source, entry, calls, **kw):
    return [sandbox.CallResult(kind="infra", error="simulated worker failure") for _ in calls]


def test_infrastructure_failure_at_submit_is_an_error_and_consumes_nothing(monkeypatch):
    task = make_task("qe_bugfix-d1", 4)
    env = QEEnvironment()
    env.reset(seed=4, task_id=task.task_id)
    env.step(QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=task.private["original"]))
    before = env.state.step_count
    monkeypatch.setattr(sandbox, "run_calls", _always_infra)
    with pytest.raises(sandbox.InfraError):
        env.step(QEAction(op="submit", answer={}))
    with pytest.raises(sandbox.InfraError):
        env.step(QEAction(op="run_tests"))
    assert not env._done and env.state.step_count == before and env._test_runs == 0
    monkeypatch.undo()
    assert env.step(QEAction(op="run_tests")).content.startswith("3/3")
    obs = env.step(QEAction(op="submit", answer={}))
    assert obs.done and obs.reward == 1.0


def test_infrastructure_failure_reaches_the_client_as_an_error_frame_not_a_reward(monkeypatch):
    from fastapi.testclient import TestClient

    from qe_arena.app import app

    task = make_task("qe_bugfix-d1", 4)
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"task_id": task.task_id, "seed": 4}}))
        assert json.loads(ws.receive_text())["type"] == "observation"
        fix = {"op": "edit", "path": MODULE, "old": task.private["buggy"], "new": task.private["original"]}
        ws.send_text(json.dumps({"type": "step", "data": fix}))  # an unedited file scores 0 without running
        assert json.loads(ws.receive_text())["type"] == "observation"
        monkeypatch.setattr(sandbox, "run_calls", _always_infra)
        ws.send_text(json.dumps({"type": "step", "data": {"op": "submit", "answer": {}}}))
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "error" and "reward" not in json.dumps(frame), frame
        monkeypatch.undo()
        ws.send_text(json.dumps({"type": "step", "data": {"op": "submit", "answer": {}}}))
        graded = json.loads(ws.receive_text())
        assert graded["type"] == "observation" and graded["data"]["done"] and graded["data"]["reward"] == 1.0
