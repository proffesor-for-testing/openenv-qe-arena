"""Flywheel one-knob search: exactly one change, a locked plan, and decisions that survive noise."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qe_flywheel import search  # noqa: E402

TEMPLATE = ROOT / "submission" / "tasks-v1.json"
# The review's simulated cells (r6): true solve rates per task, binary rewards.
P_TASK = {"qe_mutant_kill-d1": .7, "qe_mutant_kill-d2": .5, "qe_mutant_kill-d3": .2, "qe_bugfix-d2": .6,
          "qe_bugfix-d3": .3}


def config():
    return search.init_config(json.loads(TEMPLATE.read_text()))


def tasks():
    return sorted(config()["completion_tokens"])


def plan_for(predicate, cfg=None, **kwargs):
    cfg = cfg or config()
    for seed in range(2000):
        plan = search.propose(cfg, seed, set(), [], tasks(), **kwargs)
        if predicate(plan):
            return plan
    raise AssertionError("no plan matched")


def arm(split, rng, p=None, run=0, error=lambda t, s, a, r: False):
    """One independent run of a split. ``error(task, seed, attempt, reward)`` turns an episode into a
    harness error after its reward is drawn, so an arm can error selectively on its failures."""
    p = {**P_TASK, **(p or {})}
    out = []
    for t in split["task_ids"]:
        for s in split["seeds"]:
            for a in range(split["attempts"]):
                reward = float(rng.random() < p[t])
                bad = error(t, s, a, reward)
                out.append({"task_id": t, "seed": s, "attempt": a, "end": "error" if bad else "submit",
                            "reward": None if bad else reward, "seconds": run})
    return out


def decide(plan, rng, base=None, cand=None, **kw):
    return search.decide(plan, arm(plan["discovery"], rng, base, 0, **kw), arm(plan["discovery"], rng, cand, 1, **kw),
                         arm(plan["confirmation"], rng, base, 0, **kw), arm(plan["confirmation"], rng, cand, 1, **kw))


# ---------------------------------------------------------------- proposals

def test_init_config_reads_environment_and_template():
    c = config()
    assert c["max_steps"] == 12 and c["max_test_runs"] == 3
    assert c["difficulty_mix"] == {"d1": 0.2, "d2": 0.4, "d3": 0.4}


@pytest.mark.parametrize("seed", range(30))
def test_propose_changes_exactly_one_knob(seed):
    seen = set(range(830_000, 830_008))
    plan = search.propose(config(), seed, seen, [], tasks())
    assert search.changed_knobs(plan["baseline"], plan["candidate"]) == [plan["knob"]]
    assert plan["from"] != plan["to"] and plan["metric"] == "mixed_rate"
    disc, conf = set(plan["discovery"]["seeds"]), set(plan["confirmation"]["seeds"])
    assert not disc & conf and not conf & seen and not disc & seen
    assert plan == search.propose(config(), seed, seen, [], tasks())


def test_mix_change_counts_as_one_knob():
    moved = search.apply_move(config(), "difficulty_mix", {"d1": 0.1, "d2": 0.5, "d3": 0.4})
    assert search.changed_knobs(config(), moved) == ["difficulty_mix"]


def decision(knob, to, accept=False, upper=-0.1, margin=0.0):
    return {"knob": knob, "to": to, "accept": accept, "margin": margin,
            "discovery": {"upper_bound": upper, "lower_bound": upper - 0.2}}


def test_propose_blocks_only_moves_that_are_accepted_or_ruled_out():
    c = config()
    moves = search.candidate_moves(c)
    ruled_out = [decision(k, v) for k, v in moves[1:]]  # upper bound below the margin: cannot be a gain
    plan = search.propose(c, 0, set(), ruled_out, tasks())
    assert (plan["knob"], plan["to"]) == moves[0] and plan["retest"] == 0
    assert plan["rule"]["alpha"] == pytest.approx(search.ALPHA / 2)
    inconclusive = ruled_out[:-1] + [decision(*moves[-1], upper=0.08)] * 2  # rejected twice, upper bound above 0
    for seed in range(50):
        plan = search.propose(c, seed, set(), inconclusive, tasks())
        if (plan["knob"], plan["to"]) == moves[-1]:
            break
    assert (plan["knob"], plan["to"]) == moves[-1] and plan["retest"] == 2
    assert plan["rule"]["alpha"] == pytest.approx(search.ALPHA / 8)  # alpha spending: /2, /4, /8 ... sums to ALPHA
    with pytest.raises(SystemExit, match="accepted or ruled out"):
        search.propose(c, 0, set(), ruled_out + [decision(*moves[0], accept=True)], tasks())


def test_reward_metrics_never_score_a_mix_change():
    # review r6 L: a mix shift toward easy cells wins on reward without any training
    knobs = {search.propose(config(), s, set(), [], tasks(), metric="mean_reward", attempts=4,
                            discovery_seeds=8, confirm_seeds=8)["knob"] for s in range(200)}
    assert "difficulty_mix" not in knobs
    plan = plan_for(lambda p: p["knob"] == "difficulty_mix")
    plan["metric"] = "mean_reward"
    plan["plan_sha256"] = search.plan_sha(plan)
    with pytest.raises(SystemExit, match="only be scored by mixed_rate"):
        decide(plan, np.random.default_rng(0))


def test_mixed_rate_needs_fresh_instances():
    with pytest.raises(SystemExit, match="attempts 1"):
        search.propose(config(), 0, set(), [], tasks(), attempts=4)


def test_edited_plans_are_refused():
    # review r6 O / r7 O2: moving the margin after seeing data
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    edited = copy.deepcopy(plan)
    edited["rule"]["margin"] = -1.0
    with pytest.raises(SystemExit, match="plan_sha256"):
        decide(edited, np.random.default_rng(0))


# ---------------------------------------------------------------- the decision rule under noise

def test_default_plan_is_64_seeds_per_split():
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    assert len(plan["discovery"]["seeds"]) == len(plan["confirmation"]["seeds"]) == 64


def test_iid_arms_are_rarely_accepted_at_the_default_plan_size():
    # review r6 K / r7 K2 / v2 N2: the knob does nothing; both arms are independent draws of the same
    # cells, at the default 64 seeds per split. The point-estimate rule accepted 7-21% of these.
    for predicate in (lambda p: p["knob"] == "max_steps",
                      lambda p: p["knob"] == "difficulty_mix" and p["to"]["d2"] > p["from"]["d2"]):
        plan = plan_for(predicate)
        rng = np.random.default_rng(0)
        accepted = sum(decide(plan, rng)["accept"] for _ in range(100))
        assert accepted <= 5, (plan["knob"], accepted)


def test_a_mix_ratchet_toward_easy_cells_never_wins():
    # review r7 L2: repeated d3 -> d1 shifts on base-model runs (independent, same distribution)
    plan = plan_for(lambda p: p["knob"] == "difficulty_mix" and p["to"]["d1"] > p["from"]["d1"]
                    and p["to"]["d2"] == p["from"]["d2"])
    rng = np.random.default_rng(5)
    assert not any(decide(plan, rng)["accept"] for _ in range(20))


def test_leniency_that_only_makes_cells_easier_does_not_win():
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    easier = {t: min(1.0, p + 0.25) for t, p in P_TASK.items()}
    rng = np.random.default_rng(2)
    assert sum(decide(plan, rng, cand=easier)["accept"] for _ in range(40)) == 0


def test_a_real_gain_in_mixed_groups_wins_at_least_80_percent():
    # review v2 N2: three cells moved to p = .5, first test (alpha/2), 64 seeds per split
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    better = {"qe_mutant_kill-d3": .5, "qe_bugfix-d3": .5, "qe_mutant_kill-d1": .5}
    rng = np.random.default_rng(1)
    assert sum(decide(plan, rng, cand=better)["accept"] for _ in range(50)) >= 40


def test_confirmation_loss_rejects_despite_a_discovery_gain():
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(3)
    better = {"qe_mutant_kill-d3": .5, "qe_bugfix-d3": .5, "qe_mutant_kill-d1": .5}
    worse = {t: 1.0 for t in P_TASK}
    result = search.decide(plan, arm(plan["discovery"], rng, None, 0), arm(plan["discovery"], rng, better, 1),
                           arm(plan["confirmation"], rng, None, 0), arm(plan["confirmation"], rng, worse, 1))
    assert result["discovery"]["lower_bound"] > 0 and result["confirmation"]["delta"] < 0
    assert result["accept"] is False and any("confirmation" in r for r in result["reasons"])


def test_mixed_rate_estimator_skips_dropped_episodes():
    values = np.array([[1.0], [np.nan], [0.0], [1.0], [1.0]])
    codes = (values[..., None] == np.array([0.0, 1.0])).astype(float)
    got = search.task_metric(values, np.arange(5)[None, :], "mixed_rate", 4, codes)
    assert got[0] == pytest.approx(1 - math.comb(3, 4) / math.comb(4, 4))  # 4 valid: 3 ones, 1 zero -> 1.0


def test_mixed_rate_is_the_unbiased_group_estimator():
    values = np.array([[1.0], [0.0], [0.0], [0.5], [1.0], [0.0]])
    codes = np.eye(3)[np.unique(values, return_inverse=True)[1].reshape(6, 1)]
    got = search.task_metric(values, np.arange(6)[None, :], "mixed_rate", 4, codes)
    expected = 1 - (math.comb(3, 4) + math.comb(1, 4) + math.comb(2, 4)) / math.comb(6, 4)
    assert got[0] == pytest.approx(expected) == 1.0
    values = np.array([[1.0]] * 4 + [[0.0]] * 2)
    codes = np.eye(2)[np.unique(values, return_inverse=True)[1].reshape(6, 1)]
    got = search.task_metric(values, np.arange(6)[None, :], "mixed_rate", 4, codes)
    assert got[0] == pytest.approx(1 - math.comb(4, 4) / math.comb(6, 4))


# ---------------------------------------------------------------- refusals

def test_near_copies_differing_only_in_timings_are_refused():
    # review v2 m-h: the same runs with `seconds` rewritten
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(1)
    d, c = arm(plan["discovery"], rng), arm(plan["confirmation"], rng)
    with pytest.raises(SystemExit, match="same runs"):
        search.decide(plan, d, [dict(r, seconds=99) for r in d], c, [dict(r, seconds=99) for r in c])


def test_the_same_runs_in_both_arms_are_refused():
    # review r6 L: identical data in both arms used to be the documented difficulty_mix shortcut
    plan = plan_for(lambda p: p["knob"] == "difficulty_mix")
    rng = np.random.default_rng(1)
    d, c = arm(plan["discovery"], rng), arm(plan["confirmation"], rng)
    with pytest.raises(SystemExit, match="same runs"):
        search.decide(plan, d, d, c, c)


def test_error_rates_must_match_between_arms():
    # review r6 M: a candidate that errors on its hard episodes used to have them dropped pairwise
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(2)
    hard = lambda t, s, a, r: t.endswith("d3") and s % 2 == 0  # noqa: E731 - 20% of the candidate's episodes
    base = [arm(plan[k], rng, None, 0) for k in ("discovery", "confirmation")]
    cand = [arm(plan[k], rng, None, 1, error=hard) for k in ("discovery", "confirmation")]
    with pytest.raises(SystemExit, match="rates differ"):
        search.decide(plan, base[0], cand[0], base[1], cand[1])


def test_mixed_rate_drops_excluded_episodes_and_reward_metrics_count_them_as_failures():
    # review v2 N3: the trainer keeps the first 4 VALID episodes of 5, so an error is dropped, not a 0
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    split, rng = plan["discovery"], np.random.default_rng(4)
    first = lambda t, s, a, r: t == "qe_bugfix-d3" and s == split["seeds"][0]  # noqa: E731
    b, c = arm(split, rng, None, 0), arm(split, rng, None, 1, error=first)
    keyed = search.index_arm(b, split, "b"), search.index_arm(c, split, "c")
    _, cv, rates = search.arm_values(*keyed, split, plan)
    assert rates["candidate"] == pytest.approx(1 / 320) and np.isnan(cv["qe_bugfix-d3"][0, 0])
    reward_plan = dict(plan, metric="mean_reward")
    assert search.arm_values(*keyed, split, reward_plan)[1]["qe_bugfix-d3"][0, 0] == 0.0
    budget = dict(c[0], reward=0.0, end="context_budget", episode_context_tokens=12288)  # at the template limit
    assert search.arm_values(keyed[0], {**keyed[1], (budget["task_id"], budget["seed"], 0): budget}, split,
                             plan)[2]["candidate"] == pytest.approx(1 / 320)  # still only the one error


@pytest.mark.parametrize("cells, error_rate", [
    ({t: .95 for t in P_TASK}, 0.045),  # review v2: saturated cells, 4.5% random candidate errors
    ({t: .95 for t in P_TASK}, 0.03),
    (P_TASK, 0.03),
])
def test_a_candidate_that_only_errors_does_not_win(cells, error_rate):
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(4)
    noisy = lambda t, s, a, r: rng.random() < error_rate  # noqa: E731
    accepted = 0
    for _ in range(100):
        try:
            accepted += search.decide(
                plan, arm(plan["discovery"], rng, cells, 0), arm(plan["discovery"], rng, cells, 1, error=noisy),
                arm(plan["confirmation"], rng, cells, 0), arm(plan["confirmation"], rng, cells, 1, error=noisy))["accept"]
        except SystemExit:  # refused: error rates differ beyond the tolerance
            pass
    assert accepted <= 5


def test_dropping_errors_removes_their_bias_even_with_a_loose_gate():
    # with the gate opened to 10%, 4.5% random candidate errors in saturated cells: dropped, they no
    # longer manufacture mixed groups (scored as 0 they did, which is how the review's table won)
    plan = plan_for(lambda p: p["knob"] == "max_steps", error_tolerance=0.1)
    cells, rng = {t: .95 for t in P_TASK}, np.random.default_rng(8)
    noisy = lambda t, s, a, r: rng.random() < 0.045  # noqa: E731
    accepted = sum(search.decide(
        plan, arm(plan["discovery"], rng, cells, 0), arm(plan["discovery"], rng, cells, 1, error=noisy),
        arm(plan["confirmation"], rng, cells, 0), arm(plan["confirmation"], rng, cells, 1, error=noisy))["accept"]
        for _ in range(60))
    assert accepted <= 3


def test_selective_errors_under_the_tolerance_do_not_win():
    # errors placed on the candidate's FAILURES in the hard cells, at a rate inside the 2% tolerance
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(6)
    sly = lambda t, s, a, r: t.endswith("d3") and r == 0.0 and rng.random() < 0.06  # noqa: E731 - about 1.5%
    accepted = 0
    for _ in range(60):
        try:
            accepted += search.decide(
                plan, arm(plan["discovery"], rng, None, 0), arm(plan["discovery"], rng, None, 1, error=sly),
                arm(plan["confirmation"], rng, None, 0), arm(plan["confirmation"], rng, None, 1, error=sly))["accept"]
        except SystemExit:
            pass
    assert accepted <= 3


@pytest.mark.parametrize("damage, message", [
    (lambda rows: rows[:-1], "incomplete"),  # review r6 N: a single seed, task or attempt stands in for the plan
    (lambda rows: rows[:1], "incomplete"),
    (lambda rows: rows + rows[:1], "repeats"),
    (lambda rows: rows + [dict(rows[0], seed=1)], "outside the plan"),
])
def test_every_planned_episode_must_be_present(damage, message):
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(4)
    files = [arm(plan[k], rng, None, i) for k in ("discovery", "confirmation") for i in (0, 1)]
    files[1] = damage(files[1])
    with pytest.raises(SystemExit, match=message):
        search.decide(plan, files[0], files[1], files[2], files[3])


def test_arms_must_run_the_same_instances():
    plan = plan_for(lambda p: p["knob"] == "max_steps")
    rng = np.random.default_rng(4)
    files = [arm(plan[k], rng, None, i) for k in ("discovery", "confirmation") for i in (0, 1)]
    files[1][0]["corpus"] = "other_function"
    with pytest.raises(SystemExit, match="different corpus"):
        search.decide(plan, *files)


def test_mix_moves_never_take_weight_a_level_does_not_have():
    cfg = dict(config(), difficulty_mix={"d1": 0.0, "d2": 0.95, "d3": 0.05})
    mixes = [v for k, v in search.candidate_moves(cfg) if k == "difficulty_mix"]
    assert mixes and all(min(m.values()) >= 0 for m in mixes)
    assert {(k, v) for m in mixes for k, v in m.items() if v > cfg["difficulty_mix"][k]} == {("d1", 0.1), ("d3", 0.15)}
    for m in mixes:
        search.validate_config(dict(cfg, difficulty_mix=m))
