"""qe_flaky: ground truth by execution, oracle, chance-level constant policies, clean controls, repair integrity."""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import solve_actions  # noqa: E402
from qe_arena import flaky, sandbox  # noqa: E402
from qe_arena.environment import QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, make_task  # noqa: E402

LEVELS = (1, 2, 3)
MANY = 200


@pytest.fixture(scope="module")
def many():
    return {level: [make_task(f"qe_flaky-d{level}", seed) for seed in range(MANY)] for level in LEVELS}


def play(task, actions):
    env = QEEnvironment()
    obs = env.reset(seed=task.seed, task_id=task.task_id)
    texts = [obs.model_dump_json()]
    for action in actions:
        obs = env.step(QEAction(**action))
        texts.append(obs.model_dump_json())
        if obs.done:
            break
    return obs, texts


def outcomes(source, combos):
    results = sandbox.run_calls(source, "run_suite", combos)
    assert all(r.kind == "ok" for r in results), results[0]
    return [r.value for r in results]


def test_generation_is_deterministic_per_seed():
    for level in LEVELS:
        for seed in (0, 7, 123):
            a, b = make_task(f"qe_flaky-d{level}", seed), make_task(f"qe_flaky-d{level}", seed)
            assert (a.prompt, a.files, a.private["victim"], a.private["cause"]) == \
                (b.prompt, b.files, b.private["victim"], b.private["cause"])
        assert len({make_task(f"qe_flaky-d{level}", s).files[MODULE] for s in range(6)}) == 6


@pytest.mark.parametrize("level", LEVELS)
def test_ground_truth_is_what_execution_shows_on_many_seeds(many, level):
    for task in many[level]:
        p = task.private
        runs = outcomes(task.files[MODULE], p["graded"])
        varying = [n for n in p["tests"] if {r[n] for r in runs} == {True, False}]
        assert varying == ([] if p["victim"] is None else [p["victim"]]), task.seed
        assert all(all(r[n] for r in runs) for n in p["tests"] if n != p["victim"]), task.seed
        if p["victim"] is None:
            assert p["cause"] is None
            continue
        assert p["cause"] in flaky.CAUSES
        # The cause is the one facility whose per-test reset makes the flaky test's outcome constant.
        harness = p["harness"]
        boxes = re.search(r"^BOXES = \[(.*)\]$", harness, re.M).group(1).split(", ") if "BOXES = []" not in harness else []
        names = {"state": next((b for b in boxes if b in flaky.STATE_NAMES), None),
                 "store": next((b for b in boxes if b in flaky.STORE_NAMES), None)}
        probe = task.files[MODULE][:len(task.files[MODULE]) - len(harness)] + flaky._probe(harness, boxes, names)
        attributed, unmasked = flaky._attribute(probe, p["graded"], p["victim"])
        assert unmasked == runs and attributed == [p["cause"]], task.seed


@pytest.mark.parametrize("level", LEVELS)
def test_causes_and_clean_controls_are_all_represented(many, level):
    causes = [t.private["cause"] for t in many[level]]
    clean = causes.count(None) / len(causes)
    assert 0.08 <= clean <= 0.25, clean
    for cause in flaky.CAUSES:
        assert causes.count(cause) >= 15, (cause, causes.count(cause))


@pytest.mark.parametrize("level", LEVELS)
def test_oracle_earns_one(many, level):
    for task in many[level][:25]:
        obs, _ = play(task, solve_actions(task))
        assert obs.done and obs.reward == 1.0, task.seed


def _constant_policies(level):
    """Every answer that ignores the code: the k-th test in the file, with every cause, or null."""
    policies = {"null": lambda t: {"test": None}}
    for k in range(5):
        for cause in (flaky.CAUSES if level == 2 else (None,)):
            def pick(t, k=k, cause=cause):
                names = re.findall(r"^def (test_\w+)\(env\):", t.files[MODULE], re.M)
                answer = {"test": names[k] if k < len(names) else names[-1]}
                return answer | ({"cause": cause} if cause else {})
            policies[f"test#{k}" + (f"+{cause}" if cause else "")] = pick
    # Surface heuristics: the test with the most env calls, and a test that writes a module-level dict.
    policies["most_env_calls"] = lambda t: {"test": max(
        re.findall(r"^def (test_\w+)\(env\):\n((?:    .*\n?)*)", t.files[MODULE], re.M),
        key=lambda m: m[1].count("env."))[0]}
    return policies


@pytest.mark.parametrize("level", LEVELS)
def test_every_constant_policy_stays_near_chance(many, level):
    tasks = many[level]
    means = {}
    for name, policy in _constant_policies(level).items():
        means[name] = statistics.mean(flaky.grade(t, policy(t), t.files[MODULE]) for t in tasks)
    best = max(means, key=means.get)
    positional = max((k for k in means if k.startswith("test#")), key=means.get)
    print(f"\nd{level} policies over {len(tasks)} seeds: best {best} = {means[best]:.3f}; best positional "
          f"{positional} = {means[positional]:.3f}; null = {means['null']:.3f}; "
          f"most_env_calls = {means['most_env_calls']:.3f}")
    assert means[best] <= {1: 0.36, 2: 0.22, 3: 0.20}[level], means


@pytest.mark.parametrize("level", LEVELS)
def test_clean_controls_reward_only_null(many, level):
    clean = [t for t in many[level] if t.private["victim"] is None][:12]
    flaky_tasks = [t for t in many[level] if t.private["victim"] is not None][:12]
    assert clean and flaky_tasks
    for t in clean:
        assert all(all(r.values()) for r in outcomes(t.files[MODULE], t.private["graded"]))
        assert flaky.grade(t, {"test": None}, t.files[MODULE]) == 1.0
        assert flaky.grade(t, {"test": None, "cause": "shared_state"}, t.files[MODULE]) == 1.0
        for name in t.private["tests"]:
            assert flaky.grade(t, {"test": name, "cause": "shared_state"}, t.files[MODULE]) == 0.0
    for t in flaky_tasks:
        assert flaky.grade(t, {"test": None}, t.files[MODULE]) == 0.0


@pytest.mark.parametrize("answer", [{}, {"cause": "shared_state"}, {"test": 3}, {"test": ["a"]},
                                    {"test": None, "extra": 1}, "test_x", None, {"test": "test_nope"}])
def test_malformed_or_unknown_answers_earn_zero(answer):
    for level in LEVELS:
        task = next(t for s in range(50) if (t := make_task(f"qe_flaky-d{level}", s)).private["victim"])
        assert flaky.grade(task, answer, task.files[MODULE]) == 0.0


def test_d2_scores_half_for_the_test_and_full_with_the_cause(many):
    task = next(t for t in many[2] if t.private["victim"])
    p = task.private
    assert flaky.grade(task, {"test": p["victim"]}, task.files[MODULE]) == 0.5
    wrong = next(c for c in flaky.CAUSES if c != p["cause"])
    assert flaky.grade(task, {"test": p["victim"], "cause": wrong}, task.files[MODULE]) == 0.5
    assert flaky.grade(task, {"test": p["victim"], "cause": p["cause"]}, task.files[MODULE]) == 1.0
    other = next(n for n in p["tests"] if n != p["victim"])
    assert flaky.grade(task, {"test": other, "cause": p["cause"]}, task.files[MODULE]) == 0.0


def _victim_edit(task, body_lines):
    old = task.private["repair"]["old"]
    head = old.split("\n", 1)[0]
    return {"op": "edit", "path": MODULE, "old": old, "new": head + "\n" + "\n".join(body_lines)}


def _d3_tasks(many, count=12):
    return [t for t in many[3] if t.private["victim"]][:count]


@pytest.mark.parametrize("cheat", ["assert_true", "delete_asserts", "early_return", "swallow", "half_skip"])
def test_d3_repairs_that_stop_checking_earn_at_most_half(many, cheat):
    for task in _d3_tasks(many):
        body = task.private["repair"]["old"].split("\n")[1:]
        fixed = task.private["repair"]["new"].split("\n")[1:]  # skip-style cheats wrap the honest repair
        if cheat == "assert_true":
            lines = ["    assert True"]
        elif cheat == "delete_asserts":
            lines = [line for line in body if not line.strip().startswith("assert")] + ["    pass"]
        elif cheat == "early_return":
            lines = ["    return"] + fixed
        elif cheat == "swallow":
            lines = ["    try:"] + ["    " + line for line in fixed] + ["    except Exception:", "        pass"]
        else:
            lines = ["    if env.draw(2):", "        return"] + fixed
        obs, _ = play(task, [_victim_edit(task, lines), {"op": "submit", "answer": {"test": task.private["victim"]}}])
        assert obs.done and obs.reward <= 0.5, (cheat, task.seed, obs.reward)


def test_d3_naming_without_a_valid_repair_earns_half_and_the_canonical_repair_earns_one(many):
    for task in _d3_tasks(many, 6):
        victim = task.private["victim"]
        obs, _ = play(task, [{"op": "submit", "answer": {"test": victim}}])
        assert obs.reward == 0.5
        obs, _ = play(task, [{"op": "edit", "path": MODULE, **task.private["repair"]},
                             {"op": "submit", "answer": {"test": victim}}])
        assert obs.reward == 1.0
        other = next(n for n in task.private["tests"] if n != victim)
        obs, _ = play(task, [{"op": "edit", "path": MODULE, **task.private["repair"]},
                             {"op": "submit", "answer": {"test": other}}])
        assert obs.reward == 0.0


def test_d3_editing_the_harness_or_the_function_never_earns_the_repair(many):
    task = _d3_tasks(many, 1)[0]
    p = task.private
    for old, new in [("        except Exception:\n            passed[name] = False",
                      "        except Exception:\n            passed[name] = True"),
                     (task.files[MODULE].split("\n", 1)[0], task.files[MODULE].split("\n", 1)[0] + "\n    pass")]:
        obs, _ = play(task, [{"op": "edit", "path": MODULE, **p["repair"]},
                             {"op": "edit", "path": MODULE, "old": old, "new": new},
                             {"op": "submit", "answer": {"test": p["victim"]}}])
        assert obs.reward == 0.5


def test_d3_hostile_test_code_goes_through_the_static_gate(many):
    task = _d3_tasks(many, 1)[0]
    edit = _victim_edit(task, ["    import os", "    os.listdir('/')"])
    env = QEEnvironment()
    env.reset(seed=task.seed, task_id=task.task_id)
    assert not env.step(QEAction(**edit)).error
    assert env.step(QEAction(op="run_tests")).content.startswith("REJECTED")
    assert env.step(QEAction(op="submit", answer={"test": task.private["victim"]})).reward == 0.5


def test_d3_clean_control_keeps_full_credit_only_while_the_suite_passes(many):
    task = next(t for t in many[3] if t.private["victim"] is None)
    first = task.private["tests"][0]
    obs, _ = play(task, [{"op": "submit", "answer": {"test": None}}])
    assert obs.reward == 1.0
    start = task.files[MODULE].index(f"def {first}(env):\n")
    end = task.files[MODULE].index("\n\n\n", start)
    broken = f"def {first}(env):\n    assert False"
    obs, _ = play(task, [{"op": "edit", "path": MODULE, "old": task.files[MODULE][start:end], "new": broken},
                         {"op": "submit", "answer": {"test": None}}])
    assert obs.reward == 0.0


def test_edits_are_refused_below_d3():
    for level in (1, 2):
        env = QEEnvironment()
        env.reset(seed=1, task_id=f"qe_flaky-d{level}")
        assert "not available" in env.step(QEAction(op="edit", path=MODULE, old="def ", new="def ")).error


@pytest.mark.parametrize("level", LEVELS)
def test_no_ground_truth_reaches_an_observation(many, level):
    for task in many[level][:40]:
        p = task.private
        obs, texts = play(task, [{"op": "read", "path": "*"}, {"op": "run_tests"}])
        text = "\n".join(texts)
        for marker in ('"victim"', '"cause"', '"graded"', '"mutants"', '"repair"', '"private"'):
            assert marker not in text
        shown = json.loads(texts[2])["content"]
        assert all(line.endswith(f"{len(p['tests'])}/{len(p['tests'])} passed") for line in shown.splitlines())
        # The prompt is a template over the entry name, the level and the shown runs: nothing else.
        assert task.prompt == flaky._prompt(p["entry"], level, p["shown"])
        if p["victim"]:
            assert p["victim"] not in task.prompt
        region = task.files[MODULE][len(p["prefix"]):len(task.files[MODULE]) - len(p["harness"])]
        assert not any(w in region.replace(p["entry"], "").lower() for w in flaky.TELLTALE)
        for mutant in p["mutants"]:
            assert mutant["source"].split("\n", 1)[1] not in text


def test_graded_runs_are_never_shown(many):
    for level in LEVELS:
        for task in many[level][:50]:
            p = task.private
            assert {s for _, s, _ in p["shown"]}.isdisjoint({s for _, s, _ in p["graded"]})
            assert all(s < flaky.SEED_SPLIT for _, s, _ in p["shown"])
            assert len(p["graded"]) == flaky.K_GRADE


def test_run_tests_is_capped():
    env = QEEnvironment()
    env.reset(seed=3, task_id="qe_flaky-d1")
    for _ in range(3):
        assert not env.step(QEAction(op="run_tests")).error
    assert "no test runs left" in env.step(QEAction(op="run_tests")).error
