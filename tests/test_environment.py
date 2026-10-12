"""End-to-end controls on the reward channel: correct work earns 1, nothing else does."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import kill_tests, solve_actions  # noqa: E402
from qe_arena.environment import MAX_STEPS, MAX_TEST_RUNS, QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, TASK_IDS, make_task, parse_task_id  # noqa: E402

SEEDS = [11, 2026, 77_777]
PRIVATE_MARKERS = ("reference", "mutants", "private")


def play(task_id: str, seed: int, actions: list[dict]):
    env = QEEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    for action in actions:
        obs = env.step(QEAction(**action))
        if obs.done:
            break
    return env, obs


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_solution_earns_exactly_one(task_id, seed):
    _, obs = play(task_id, seed, solve_actions(make_task(task_id, seed)))
    assert obs.done and obs.reward == 1.0


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("seed", SEEDS)
def test_doing_nothing_earns_zero(task_id, seed):
    _, obs = play(task_id, seed, [{"op": "read", "path": "*"}, {"op": "submit", "answer": {}}])
    assert obs.done and obs.reward == 0.0


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_same_seed_reproduces_the_task_and_different_seeds_vary(task_id):
    a, b = make_task(task_id, 5), make_task(task_id, 5)
    assert (a.prompt, a.files) == (b.prompt, b.files)
    variants = {json.dumps(make_task(task_id, s).files, sort_keys=True) for s in range(6)}
    assert len(variants) >= 4


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_observations_never_carry_private_verifier_data(task_id):
    task = make_task(task_id, 3)
    env = QEEnvironment()
    texts = [env.reset(seed=3, task_id=task_id).model_dump_json()]
    texts.append(env.step(QEAction(op="read", path="*")).model_dump_json())
    for marker in PRIVATE_MARKERS:
        assert all(f'"{marker}"' not in t for t in texts)
    if task.family == "qe_bugfix":
        assert task.private["original"] not in texts[1]
    else:
        for m in task.private["mutants"]:
            assert m["source"] not in texts[1]


def test_prompt_states_the_reply_format_with_concrete_actions():
    for task_id in TASK_IDS:
        prompt = make_task(task_id, 1).prompt
        assert "exactly one JSON action object" in prompt
        assert '{"op": "read", "path": "*"}' in prompt
        assert '"op": "submit"' in prompt


@pytest.mark.parametrize("level", [1, 2, 3])
def test_bugfix_partial_or_wrong_fix_earns_zero(level):
    task = make_task(f"qe_bugfix-d{level}", 8)
    env = QEEnvironment()
    env.reset(seed=8, task_id=task.task_id)
    # A cosmetic change is not a fix.
    obs = env.step(QEAction(op="edit", path=MODULE, old=task.files[MODULE], new=task.files[MODULE] + "\n# fixed\n"))
    assert not obs.error
    assert env.step(QEAction(op="submit")).reward == 0.0


@pytest.mark.parametrize("task_id", ["qe_bugfix-d2", "qe_bugfix-d3"])
def test_bugfix_leaving_any_single_fault_unfixed_earns_zero(task_id):
    from qe_arena import mutation

    import itertools

    task = make_task(task_id, 21)
    original = task.private["original"]
    # Descriptions omit the column and can repeat on one line, so recover the exact sites by reproducing the file.
    # Real faults are recorded sites, not operator sites: include the recorded ones.
    pool = dict.fromkeys([*mutation.enumerate_sites(original), *task.private["fault_sites"]])
    candidates = [s for s in pool if s.description in task.private["sites"]]
    sites = None
    for combo in itertools.combinations(candidates, len(task.private["sites"])):
        try:
            if mutation.apply(original, list(combo)) == task.private["buggy"]:
                sites = list(combo)
                break
        except ValueError:
            continue
    assert sites is not None and len(sites) == int(task_id[-1])
    for site in sites:  # every other fault fixed, this one left in
        _, obs = play(task_id, 21, [
            {"op": "edit", "path": MODULE, "old": task.private["buggy"], "new": mutation.apply(original, [site])},
            {"op": "submit", "answer": {}},
        ])
        assert obs.reward == 0.0, site.description


@pytest.mark.parametrize("payload", [
    "def {entry}(*a, **k):\n    return open('/app/qe_arena/corpus/x.py').read()\n",
    "import os\ndef {entry}(*a, **k):\n    return os.listdir('/')\n",
    "def {entry}(*a, **k):\n    return ().__class__.__mro__[1].__subclasses__()\n",
    "def {entry}(*a, **k):\n    return __import__('qe_arena').tasks\n",
    "def not_the_entry(*a):\n    return 0\n",
    "def {entry}(*a, **k):\n    while True:\n        pass\n",
])
def test_bugfix_hostile_or_broken_code_earns_zero(payload):
    task = make_task("qe_bugfix-d1", 4)
    code = payload.format(entry=task.private["entry"])
    _, obs = play(task.task_id, 4, [
        {"op": "edit", "path": MODULE, "old": task.private["buggy"], "new": code},
        {"op": "submit", "answer": {}},
    ])
    assert obs.done and obs.reward == 0.0


def test_bugfix_hardcoding_the_visible_tests_earns_zero():
    task = make_task("qe_bugfix-d1", 12)
    entry, visible = task.private["entry"], task.private["visible"]
    table = {repr(v["args"]): v["expected"].value for v in visible if v["expected"].kind == "ok"}
    # Passes the static gate (no imports, no banned names): only the hidden tests stop it.
    code = f"TABLE = {table!r}\ndef {entry}(*args):\n    return TABLE.get(repr(list(args)))\n"
    _, obs = play(task.task_id, 12, [
        {"op": "edit", "path": MODULE, "old": task.private["buggy"], "new": code},
        {"op": "submit", "answer": {}},
    ])
    assert obs.reward == 0.0


def test_run_tests_reports_and_is_capped():
    task = make_task("qe_bugfix-d1", 9)
    env = QEEnvironment()
    env.reset(seed=9, task_id=task.task_id)
    first = env.step(QEAction(op="run_tests"))
    assert first.content.startswith("3/3 visible tests passed")  # visible tests pass; the fault hides
    for _ in range(MAX_TEST_RUNS - 1):
        assert not env.step(QEAction(op="run_tests")).error
    assert "no test runs left" in env.step(QEAction(op="run_tests")).error


def test_edit_requires_exactly_one_occurrence_and_the_module():
    task = make_task("qe_bugfix-d1", 9)
    env = QEEnvironment()
    env.reset(seed=9, task_id=task.task_id)
    assert "exactly once" in env.step(QEAction(op="edit", path=MODULE, old="zzz-not-there", new="x")).error
    assert "can be edited" in env.step(QEAction(op="edit", path="other.py", old="a", new="b")).error
    assert "non-empty" in env.step(QEAction(op="edit", path=MODULE, old="", new="b")).error


def test_bugfix_only_ops_are_refused_in_mutant_kill():
    env = QEEnvironment()
    env.reset(seed=1, task_id="qe_mutant_kill-d1")
    assert "not available" in env.step(QEAction(op="run_tests")).error


@pytest.mark.parametrize("level", [1, 2, 3])
def test_mutant_kill_wrong_expected_earns_half(level):
    task = make_task(f"qe_mutant_kill-d{level}", 30)
    tests = kill_tests(task, wrong_expected=True)
    _, obs = play(task.task_id, 30, [{"op": "submit", "answer": {"tests": tests}}])
    assert obs.reward == pytest.approx(0.5)


def test_mutant_kill_correct_expected_on_a_non_killing_input_earns_zero():
    task = make_task("qe_mutant_kill-d1", 31)
    p = task.private
    from qe_arena import sandbox

    outputs = sandbox.run_calls(p["mutants"][0]["source"], p["entry"], p["inputs"])
    index = next(i for i, (r, o) in enumerate(zip(p["reference"], outputs)) if sandbox.compare(r, o))
    ref = p["reference"][index]
    expected = {"raises": ref.value} if ref.kind == "exc" else ref.value
    _, obs = play(task.task_id, 31, [
        {"op": "submit", "answer": {"tests": [{"args": p["inputs"][index], "expected": expected}]}}])
    assert obs.reward == 0.0


def test_mutant_kill_d3_partial_kills_score_proportionally():
    task = make_task("qe_mutant_kill-d3", 40)
    tests = kill_tests(task)
    _, obs = play(task.task_id, 40, [{"op": "submit", "answer": {"tests": tests[:1]}}])
    assert 1 / 3 - 1e-9 <= obs.reward <= 1.0  # the first test kills at least its own mutant


@pytest.mark.parametrize("answer", [
    {"tests": "nope"},
    {"tests": [{"args": [1]}]},
    {"tests": [{"args": "x", "expected": 1}]},
    {"tests": [{"args": [1], "expected": 1, "extra": 2}]},
    {"tests": [{"args": [1], "expected": 1}] * 2},
    {"tests": [{"args": ["x" * 5000], "expected": 1}]},
])
def test_mutant_kill_malformed_answers_earn_zero(answer):
    _, obs = play("qe_mutant_kill-d1", 5, [{"op": "submit", "answer": answer}])
    assert obs.done and obs.reward == 0.0


def test_step_budget_ends_the_episode_with_zero():
    env = QEEnvironment()
    env.reset(seed=1, task_id="qe_mutant_kill-d2")
    obs = None
    for _ in range(MAX_STEPS):
        obs = env.step(QEAction(op="read", path="*"))
    assert obs.done and obs.reward == 0.0
    assert "terminal" in env.step(QEAction(op="read", path="*")).error


@pytest.mark.parametrize("bad", ["qe_bugfix", "qe_bugfix-d4", "other-d1", "qe_bugfix-d1 ", 7])
def test_invalid_task_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_task_id(bad)
