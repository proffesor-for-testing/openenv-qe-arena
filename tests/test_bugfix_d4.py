"""qe_bugfix-d4: d3's faults plus a total edit budget, so the faults must be localised."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import solve_actions  # noqa: E402
from qe_arena import edit_budget, mutation, tasks  # noqa: E402
from qe_arena.environment import MAX_FREE_REFUSALS, MAX_STEPS, QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, make_task, parse_task_id  # noqa: E402

D4 = "qe_bugfix-d4"


def _env(seed):
    env = QEEnvironment()
    env.reset(seed=seed, task_id=D4)
    return env


def test_d4_is_a_bugfix_only_cell_appended_after_the_original_six():
    assert tasks.TASK_IDS[:6] == [f"{f}-d{lv}" for f in tasks.FAMILIES for lv in (1, 2, 3)]
    assert tasks.TASK_IDS[6:] == [D4]
    assert parse_task_id(D4) == ("qe_bugfix", 4)
    for bad in ("qe_mutant_kill-d4", "qe_bugfix-d5", "qe_bugfix-d0"):
        with pytest.raises(ValueError):
            parse_task_id(bad)
    assert tasks.format_task_id("qe_bugfix", 4) == D4
    with pytest.raises(ValueError):
        tasks.format_task_id("qe_mutant_kill", 4)
    assert tasks.FAULTS[4] == tasks.FAULTS[3] and tasks.MIN_SUBTLE[4] == tasks.MIN_SUBTLE[3]


def test_oracle_earns_one_within_budget_and_steps_on_200_seeds():
    for seed in range(200):
        task = make_task(D4, seed)
        p = task.private
        budget = p["edit_budget"]
        fix = edit_budget.minimal_fix(p["buggy"], p["original"])
        assert sum(edit_budget.edit_cost(e) for e in fix) <= budget, seed  # the minimal fix fits
        assert len(p["buggy"]) + len(p["original"]) > budget, seed  # a full rewrite does not
        assert mutation.apply(p["original"], p["fault_sites"]) == p["buggy"]
        env = QEEnvironment()
        obs = env.reset(seed=seed, task_id=D4)
        actions = solve_actions(task)
        assert len(actions) <= MAX_STEPS
        for action in actions:
            obs = env.step(QEAction(**action))
            assert not obs.error, (seed, obs.error)
            if obs.done:
                break
        assert obs.done and obs.reward == 1.0, seed


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_full_rewrite_is_refused_unapplied_and_free(seed):
    task = make_task(D4, seed)
    env = _env(seed)
    obs = env.step(QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=task.private["original"]))
    assert obs.error.startswith("edit refused") and env._files[MODULE] == task.private["buggy"]
    assert env.state.step_count == 0 and obs.remaining_steps == MAX_STEPS
    assert f"of the {task.private['edit_budget']} budget are left" in obs.error


def test_budget_arithmetic_is_exact():
    for length in (0, 1, 124, 125, 126, 1000, 1001):
        raw = math.ceil(edit_budget.BUDGET_SHARE * length)
        expected = -(-raw // edit_budget.BUDGET_QUANTUM) * edit_budget.BUDGET_QUANTUM
        assert edit_budget.budget_for("x" * length) == expected
        assert expected % edit_budget.BUDGET_QUANTUM == 0 and expected >= raw
    # Spending exactly the budget is allowed; one character more is refused; the reply shows what is left.
    task = make_task(D4, 3)
    budget, text = task.private["edit_budget"], task.private["buggy"]
    line = next(ln for ln in text.splitlines(keepends=True) if text.count(ln) == 1 and ln.strip())
    pad = budget - 2 * len(line)
    assert pad >= 0
    over = _env(3).step(QEAction(op="edit", path=MODULE, old=line, new=line + "#" * (pad + 1)))
    assert over.error.startswith("edit refused: it costs") and f"{budget} of the {budget} budget" in over.error
    env = _env(3)
    exact = env.step(QEAction(op="edit", path=MODULE, old=line, new=line + "#" * pad))
    assert not exact.error and exact.content.endswith(f"edit budget left: 0 of {budget} characters")
    after = env.step(QEAction(op="edit", path=MODULE, old=line + "#" * pad, new=line))
    assert after.error.startswith("edit refused") and f"0 of the {budget} budget" in after.error


@pytest.mark.parametrize("seed", range(6))
def test_the_budget_reveals_nothing_about_the_faults(seed):
    """The shown number is a function of the shown file alone: equal for every fault set of that length."""
    task = make_task(D4, seed)
    budget = task.private["edit_budget"]
    assert budget == edit_budget.budget_for(task.files[MODULE])
    assert task.prompt.count(str(budget)) == 1 and f"Edit budget: {budget} characters" in task.prompt
    # Any other fault set on the same original with the same buggy length would show the same budget.
    other = "x" * len(task.files[MODULE])
    assert edit_budget.budget_for(other) == budget


def test_refused_edits_are_free_up_to_a_cap_so_episodes_stay_bounded():
    task = make_task(D4, 4)
    env = _env(4)
    rewrite = QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=task.private["original"])
    for _ in range(MAX_FREE_REFUSALS):
        assert env.step(rewrite).error.startswith("edit refused")
    assert env.state.step_count == 0
    env.step(rewrite)
    assert env.state.step_count == 1  # past the cap a refusal uses a step


def test_lower_levels_carry_no_budget():
    for level in (1, 2, 3):
        task = make_task(f"qe_bugfix-d{level}", 0)
        assert task.private["edit_budget"] is None and "Edit budget" not in task.prompt


def test_d4_uses_its_own_random_stream():
    assert make_task(D4, 7).prompt != make_task("qe_bugfix-d3", 7).prompt
    assert random.Random(f"{D4}:7").random() != random.Random("qe_bugfix-d3:7").random()
