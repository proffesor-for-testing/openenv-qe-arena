"""Tests that kill surviving verifier mutants (proposed additions to tests/test_reward_integrity.py)."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qe_arena import corpus, mutation, sandbox, tasks
from qe_arena.environment import MODULE, QEAction, QEEnvironment
from qe_arena.tasks import Task, grade_bugfix, grade_mutant_kill, make_task, parse_task_id

SUBTLE = 0.15  # spec literal: tasks.SUBTLE_RATE may itself be the mutant


def kill_task(original: str, mutants: list[str], difficulty: int = 1, entry: str = "f") -> Task:
    return Task(task_id=f"qe_mutant_kill-d{difficulty}", family="qe_mutant_kill", difficulty=difficulty,
                seed=0, prompt="", files={},
                private={"original": original, "entry": entry,
                         "mutants": [{"source": m, "site": "", "random_kill_rate": 0.5} for m in mutants]})


ORIG = ("def f(x):\n"
        "    if x < 0:\n        raise ValueError('neg')\n"
        "    if x == 7:\n        return {1}\n"          # unserializable -> crash in both programs
        "    if x == 9:\n        return 2.0\n"
        "    if x == 11:\n        return 1.0\n"
        "    return x + 1\n")
MUT_PLUS = ORIG.replace("return x + 1", "return x + 2").replace("return 2.0", "return 3.0").replace(
    "return 1.0", "return 0.0").replace("raise ValueError('neg')", "raise TypeError('neg')")


# ---------------------------------------------------------------- grade_mutant_kill / _expected_matches

def test_input_where_original_does_not_return_cleanly_earns_nothing():
    # Kills M-gmk-drop-refkind: crash/timeout on the ORIGINAL is not a distinguishing input.
    task = kill_task(ORIG, [ORIG.replace("return x + 1", "return x + 2")])
    assert grade_mutant_kill(task, {"tests": [{"args": [7], "expected": None}]}) == 0.0


def test_reward_is_the_mean_over_mutants_not_the_best():
    # Kills M-gmk-max-scores / M-gmk-min-scores: one of three mutants killed exactly -> 1/3.
    m1 = ORIG.replace("return x + 1", "return x + 2")
    m2 = ORIG.replace("if x == 9:", "if x == 99:")
    m3 = ORIG.replace("if x == 11:", "if x == 111:")
    task = kill_task(ORIG, [m1, m2, m3], difficulty=3)
    assert grade_mutant_kill(task, {"tests": [{"args": [3], "expected": 4}]}) == pytest.approx(1 / 3)
    both = {"tests": [{"args": [3], "expected": 4}, {"args": [9], "expected": 2.0}]}
    assert grade_mutant_kill(task, both) == pytest.approx(2 / 3)


def test_several_killing_tests_for_one_mutant_do_not_add_up():
    # Kills M-gmk-sum-not-max at d3 (3 tests allowed): credit per mutant is capped at 1.
    m1 = ORIG.replace("return x + 1", "return x + 2")
    task = kill_task(ORIG, [m1], difficulty=3)
    tests = [{"args": [n], "expected": n + 1} for n in (1, 2, 3)]
    assert grade_mutant_kill(task, {"tests": tests}) == 1.0


def test_wrong_exception_name_earns_only_half():
    # Kills M-em-any-exc-name and M-em-raises-subset.
    task = kill_task(ORIG, [MUT_PLUS])
    assert grade_mutant_kill(task, {"tests": [{"args": [-1], "expected": {"raises": "ValueError"}}]}) == 1.0
    assert grade_mutant_kill(task, {"tests": [{"args": [-1], "expected": {"raises": "TypeError"}}]}) == 0.5
    assert grade_mutant_kill(task, {"tests": [{"args": [-1], "expected": {"raises": "ValueError", "x": 1}}]}) == 0.5


def test_integer_spelling_of_a_float_result_is_exact_but_bool_is_not():
    # Kills M-em-drop-intfloat (2 for 2.0 must stay exact) and M-em-bool-coerce (True is not 1.0).
    task = kill_task(ORIG, [MUT_PLUS])
    assert grade_mutant_kill(task, {"tests": [{"args": [9], "expected": 2}]}) == 1.0
    assert grade_mutant_kill(task, {"tests": [{"args": [11], "expected": True}]}) == 0.5
    assert grade_mutant_kill(task, {"tests": [{"args": [3], "expected": 4.0}]}) == 0.5  # int result stays strict


def _try_apply(source, sites):
    try:
        return mutation.apply(source, sites)
    except ValueError:
        return None


# ---------------------------------------------------------------- generator invariants (reward calibration)

SEEDS = [*range(6), 23]  # 23: a d3 seed where an equivalent-alone fault is reachable


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", SEEDS)
def test_mutant_kill_mutants_meet_their_level_contract(level, seed):
    # Kills M-mk-drop-rate0, M-mk-d1-rate-0, M-mk-d23-drop-subtle, M-subtle-rate-*, range_bound on names.
    task = make_task(f"qe_mutant_kill-d{level}", seed)
    p = task.private
    assert len(p["mutants"]) == (3 if level == 3 else 1)
    assert sum(name.endswith(".diff") for name in task.files) == len(p["mutants"])
    for m in p["mutants"]:
        outputs = sandbox.run_calls(m["source"], p["entry"], p["inputs"])
        rate = sum(not sandbox.compare(r, o) for r, o in zip(p["reference"], outputs)) / len(p["inputs"])
        assert rate == m["random_kill_rate"] and rate > 0
        assert rate >= 0.25 if level == 1 else rate <= SUBTLE


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", SEEDS)
def test_bugfix_task_contract(level, seed):
    # Kills M-bf-killed-single, M-bf-visible-any, M-bf-d3-drop-subtle, M-bf-subtle-latch, M-bf-drop-single,
    # M-bf-report-wrong-input, M-usable-any.
    task = make_task(f"qe_bugfix-d{level}", seed)
    p = task.private
    assert all(r.kind in ("ok", "exc") for r in p["reference"])
    buggy_out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    failing = [i for i, (r, o) in enumerate(zip(p["reference"], buggy_out)) if not sandbox.compare(r, o)]
    assert failing, "the buggy file must fail some hidden input"
    for v in p["visible"]:  # every visible test passes on the buggy file
        [got] = sandbox.run_calls(p["buggy"], p["entry"], [v["args"]])
        assert sandbox.compare(v["expected"], got)
    # Descriptions omit the column, so resolve the exact sites by reproducing the buggy file.
    pool = dict.fromkeys([*mutation.enumerate_sites(p["original"]), *p["fault_sites"]])  # real faults too
    candidates = [s for s in pool if s.description in p["sites"]]
    want = tasks.FAULTS[level]
    assert len(p["sites"]) == want
    combos = [list(c) for c in itertools.combinations(candidates, want)]
    sites = next(c for c in combos if _try_apply(p["original"], c) == p["buggy"])
    rates = []
    for s in sites:  # each fault alone changes behaviour
        out = sandbox.run_calls(mutation.apply(p["original"], [s]), p["entry"], p["inputs"])
        rates.append(sum(not sandbox.compare(r, o) for r, o in zip(p["reference"], out)) / len(p["inputs"]))
    assert min(rates) > 0
    assert sum(r <= SUBTLE for r in rates) >= tasks.MIN_SUBTLE[level]
    if level == 1:  # the report names an input that reveals the bug and is never graded
        reported = p["report_args"]
        assert tasks._call_text(p["entry"], reported) in task.prompt
        [want_r] = sandbox.run_calls(p["original"], p["entry"], [reported])
        [got_r] = sandbox.run_calls(p["buggy"], p["entry"], [reported])
        assert not sandbox.compare(want_r, got_r)
        assert json.dumps(reported, sort_keys=True) not in {json.dumps(a, sort_keys=True) for a in p["inputs"]}


def test_unknown_family_with_qe_prefix_is_rejected():
    # Kills M-parse-drop-family.
    with pytest.raises(ValueError):
        parse_task_id("qe_other-d1")


# ---------------------------------------------------------------- environment


def test_submit_is_terminal_and_cannot_be_retried():
    # Kills M-env-submit-not-done: best-of-n resubmission against the hidden oracle.
    env = QEEnvironment()
    env.reset(seed=1, task_id="qe_mutant_kill-d1")
    first = env.step(QEAction(op="submit", answer={"tests": [{"args": [], "expected": 0}]}))
    assert first.done
    again = env.step(QEAction(op="submit", answer={}))
    assert again.done and "terminal" in again.error and again.reward == first.reward


def test_non_finite_or_oversize_answers_score_zero_without_crashing():
    # Kills M-env-drop-except, M-env-allow-nan, M-env-drop-answercap.
    from oracle import kill_tests  # noqa: PLC0415
    task = make_task("qe_mutant_kill-d1", 5)
    good = kill_tests(task)
    for answer in ({"tests": good, "pad": float("nan")}, {"tests": good, "pad": "x" * 9000}):
        env = QEEnvironment()
        env.reset(seed=5, task_id=task.task_id)
        obs = env.step(QEAction(op="submit", answer=answer))
        assert obs.done and obs.reward == 0.0
    env = QEEnvironment()
    env.reset(seed=5, task_id=task.task_id)
    assert env.step(QEAction(op="submit", answer={"tests": good})).reward == 1.0  # control


def test_edit_caps_and_ambiguity_are_errors():
    # Kills M-env-count-zero, M-env-drop-editcap, M-env-drop-filecap.
    task = make_task("qe_bugfix-d1", 9)
    env = QEEnvironment()
    env.reset(seed=9, task_id=task.task_id)
    text = task.files[MODULE]
    assert text.count("    ") > 1
    assert "exactly once" in env.step(QEAction(op="edit", path=MODULE, old="    ", new="  ")).error
    unique = [ln for ln in dict.fromkeys(text.splitlines(keepends=True)) if text.count(ln) == 1 and ln.strip()]
    a, b = unique[0], unique[1]
    assert "edit small" in env.step(QEAction(op="edit", path=MODULE, old=a, new=a + "#" * 4096)).error
    pad = "#" * 3500 + "\n"
    assert not env.step(QEAction(op="edit", path=MODULE, old=a, new=a + pad)).error
    assert "too large" in env.step(QEAction(op="edit", path=MODULE, old=b, new=b + pad)).error


# ---------------------------------------------------------------- round 2: remaining survivors

LEN_ORIG = "def f(s):\n    if len(s) > 3000:\n        return -1\n    return len(s)\n"
LEN_MUT = LEN_ORIG.replace("return len(s)\n", "return len(s) + 1\n")


def _sized_test(total_bytes: int) -> dict:
    """A killing test whose compact JSON is exactly ``total_bytes`` long."""
    def build(n):
        return {"args": ["x" * n], "expected": n}
    n = total_bytes - len(json.dumps(build(0), separators=(",", ":")))
    while len(json.dumps(build(n), separators=(",", ":"))) > total_bytes:
        n -= 1
    test = build(n)
    assert len(json.dumps(test, separators=(",", ":"))) == total_bytes
    return test


def test_per_test_size_cap_is_inclusive_and_enforced():
    # Kills E-tasks-289 (> -> >=) and M-gmk-drop-argsbytes.
    task = kill_task(LEN_ORIG, [LEN_MUT])
    assert grade_mutant_kill(task, {"tests": [_sized_test(tasks.MAX_ARGS_BYTES)]}) == 1.0
    assert grade_mutant_kill(task, {"tests": [_sized_test(tasks.MAX_ARGS_BYTES + 1)]}) == 0.0


def test_test_count_cap_and_shape_hold_even_when_a_test_kills():
    # Kills M-gmk-drop-maxtests, M-gmk-maxtests-d3, E-tasks-283 (or->and), M-gmk-extra-keys.
    task = kill_task(LEN_ORIG, [LEN_MUT], difficulty=1)
    killing = {"args": ["ab"], "expected": 2}
    assert grade_mutant_kill(task, {"tests": [killing]}) == 1.0
    assert grade_mutant_kill(task, {"tests": [killing, killing]}) == 0.0
    assert grade_mutant_kill(task, {"tests": [dict(killing, note=1)]}) == 0.0
    d2 = kill_task(LEN_ORIG, [LEN_MUT], difficulty=2)
    assert grade_mutant_kill(d2, {"tests": [killing, killing]}) == 0.0


def test_unhashable_or_too_deep_expected_never_earns_full_credit():
    # Kills E-sandbox-565 (compare's except branch returning True): an expected value too deep to
    # normalize must not be treated as a match.
    task = kill_task(LEN_ORIG, [LEN_MUT])
    deep = 0
    for _ in range(1010):
        deep = [deep]
    test = {"args": ["ab"], "expected": deep}
    assert len(json.dumps(test, separators=(",", ":"))) <= tasks.MAX_ARGS_BYTES
    assert grade_mutant_kill(task, {"tests": [test]}) == 0.5


def test_bugfix_grade_requires_every_hidden_input():
    # Kills M-gb-threshold (skipping the last hidden input) and M-gb-all-any.
    original = "def f(x):\n    return x + 1\n"
    inputs = [[i] for i in range(10)]
    task = Task(task_id="qe_bugfix-d1", family="qe_bugfix", difficulty=1, seed=0, prompt="", files={},
                private={"entry": "f", "original": original, "buggy": "def f(x):\n    return x\n",
                         "inputs": inputs, "reference": sandbox.run_calls(original, "f", inputs)})
    assert grade_bugfix(task, original) == 1.0
    assert grade_bugfix(task, "def f(x):\n    return x + 1 if x != 9 else 0\n") == 0.0
    assert grade_bugfix(task, "def f(x):\n    return x + 1 if x != 0 else 0\n") == 0.0


def test_compare_dicts_with_same_size_different_keys():
    # Kills M-eq-dict-keys (KeyError escapes compare instead of a clean mismatch).
    assert sandbox.compare(sandbox.CallResult("ok", {"a": 1}), sandbox.CallResult("ok", {"b": 1})) is False


def test_static_gate_depth_limit_is_inclusive():
    # Kills E-sandbox-189 (> -> >=).
    def src(k):
        return "def f():\n    return " + "-" * k + "1\n"
    k = next(k for k in range(200) if sandbox._ast_depth(__import__("ast").parse(src(k))) == sandbox.MAX_AST_DEPTH)
    assert sandbox.static_gate(src(k)) == []
    assert sandbox.static_gate(src(k + 1))


def test_reads_show_files_and_the_step_budget_counts_one_per_step():
    # Kills E-environment-85, E-environment-87, E-environment-77 (+= 2).
    task = make_task("qe_bugfix-d1", 9)
    env = QEEnvironment()
    env.reset(seed=9, task_id=task.task_id)
    everything = env.step(QEAction(op="read", path="*"))
    assert task.files[MODULE] in everything.content and not everything.error
    one = env.step(QEAction(op="read", path=MODULE))
    assert one.content == task.files[MODULE] and not one.error
    assert "unknown file" in env.step(QEAction(op="read", path="nope.py")).error
    from qe_arena.environment import MAX_STEPS  # noqa: PLC0415
    for _ in range(MAX_STEPS - 4):
        obs = env.step(QEAction(op="read", path=MODULE))
    assert not obs.done and obs.remaining_steps == 1
    assert env.step(QEAction(op="submit")).done


def test_edit_and_answer_caps_are_inclusive():
    # Kills E-environment-102 (>=, and + -> -), E-environment-109 (>=), E-environment-121 (>=).
    from oracle import kill_tests  # noqa: PLC0415
    from qe_arena.environment import MAX_ANSWER_BYTES, MAX_EDIT_BYTES, MAX_FILE_BYTES  # noqa: PLC0415
    task = make_task("qe_bugfix-d1", 9)
    env = QEEnvironment()
    env.reset(seed=9, task_id=task.task_id)
    text = task.files[MODULE]
    a = next(ln for ln in text.splitlines(keepends=True) if text.count(ln) == 1 and ln.strip())
    obs = env.step(QEAction(op="edit", path=MODULE, old=a, new=a + "#" * (MAX_EDIT_BYTES - 2 * len(a)) ))
    assert not obs.error  # exactly MAX_EDIT_BYTES is allowed (file stays under the file cap)
    current = env._files[MODULE]
    b = next(ln for ln in current.splitlines(keepends=True) if current.count(ln) == 1 and ln.strip() and "#" not in ln)
    room = MAX_FILE_BYTES - len(current.encode())
    assert 0 < room <= MAX_EDIT_BYTES - 2 * len(b)
    assert not env.step(QEAction(op="edit", path=MODULE, old=b, new=b + "#" * room)).error
    assert len(env._files[MODULE].encode()) == MAX_FILE_BYTES
    kt = make_task("qe_mutant_kill-d1", 5)
    good = kill_tests(kt)
    base = len(json.dumps({"tests": good, "pad": ""}, separators=(",", ":")).encode())
    answer = {"tests": good, "pad": "x" * (MAX_ANSWER_BYTES - base)}
    assert len(json.dumps(answer, separators=(",", ":")).encode()) == MAX_ANSWER_BYTES
    env = QEEnvironment()
    env.reset(seed=5, task_id=kt.task_id)
    assert env.step(QEAction(op="submit", answer=answer)).reward == 1.0


GOLDEN = {
    "qe_bugfix-d1:0": "7816de45ee0c6b6c", "qe_bugfix-d1:1": "24559b6fa19b3a65",
    "qe_bugfix-d1:2": "c485566cf41da3f6", "qe_bugfix-d1:3": "7637736f4a4b27e9",
    "qe_bugfix-d2:0": "793c43825074f5ef", "qe_bugfix-d2:1": "2eb2deb692c9c106",
    "qe_bugfix-d2:2": "af8819b151c56e35", "qe_bugfix-d2:3": "3a4e674811293e8f",
    "qe_bugfix-d3:0": "9a179b75b06d8c3c", "qe_bugfix-d3:1": "694a306e26f85371",
    "qe_bugfix-d3:2": "baf3006a24382d1e", "qe_bugfix-d3:3": "1ec2c7078bb50b63",
    "qe_bugfix-d4:0": "a6a5965767672da6", "qe_bugfix-d4:1": "c1e2e89e18611c73",
    "qe_bugfix-d4:2": "c0a2e4755ca9f9ce", "qe_bugfix-d4:3": "88eb2d2de76e55bf",
    "qe_mutant_kill-d1:0": "2c18117ee463689c", "qe_mutant_kill-d1:1": "8bfe4fb2a811ab6b",
    "qe_mutant_kill-d1:2": "a29fa8404c66b6ff", "qe_mutant_kill-d1:3": "7579e245d91a9739",
    "qe_mutant_kill-d2:0": "0b352307ed8023d3", "qe_mutant_kill-d2:1": "1ea3d0f1e672b099",
    "qe_mutant_kill-d2:2": "df7e55e364e1dac7", "qe_mutant_kill-d2:3": "ecfc7229e35dcbaa",
    "qe_mutant_kill-d3:0": "cde1438f42f6d784", "qe_mutant_kill-d3:1": "b639dafcf5ee48c1",
    "qe_mutant_kill-d3:2": "bfbcc2137649b001", "qe_mutant_kill-d3:3": "39acd34c29d9620f",
}


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_task_generation_is_pinned(key):
    # Snapshot: any change to task selection (thresholds, loop control, prompts) must be deliberate.
    # Regenerate deliberately when the generator changes: sha256(json.dumps([prompt, files], sort_keys=True))[:16].
    import hashlib  # noqa: PLC0415
    task_id, seed = key.split(":")
    t = make_task(task_id, int(seed))
    assert hashlib.sha256(json.dumps([t.prompt, t.files], sort_keys=True).encode()).hexdigest()[:16] == GOLDEN[key]


def test_raises_expectation_needs_an_actual_exception():
    # Kills M-em-drop-kind-exc: a returned string that happens to name an exception is not a raise.
    original = "def f(x):\n    if x == 5:\n        return 'ValueError'\n    return 'ok'\n"
    task = kill_task(original, [original.replace("x == 5", "x == 6")])
    assert grade_mutant_kill(task, {"tests": [{"args": [5], "expected": "ValueError"}]}) == 1.0
    assert grade_mutant_kill(task, {"tests": [{"args": [5], "expected": {"raises": "ValueError"}}]}) == 0.5
