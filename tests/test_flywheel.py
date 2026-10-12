"""Flywheel stats and select: posterior math, grouping, excluded rows, and cell selection.

Ledger tests are in test_flywheel_ledger.py, one-knob search in test_flywheel_search.py."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flywheel as cli  # noqa: E402
from qe_flywheel import stats  # noqa: E402
from qe_flywheel import select as sel  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "flywheel"
TEMPLATE = ROOT / "submission" / "tasks-v1.json"


def row(task_id, seed, attempt, reward, corpus=None, **extra):
    r = {"task_id": task_id, "seed": seed, "attempt": attempt, "reward": reward, "end": "submit",
         "steps": 4, "invalid": 0}
    if corpus is not None:
        r["corpus"] = corpus
    if reward is None:
        r["end"] = "error"
    r.update(extra)
    return r


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ---------------------------------------------------------------- posterior math against closed forms

@pytest.mark.parametrize("x", [0.01, 0.1, 0.37, 0.5, 0.8, 0.99])
def test_beta_cdf_matches_closed_forms(x):
    assert stats.beta_cdf(x, 1, 1) == pytest.approx(x, abs=1e-12)
    assert stats.beta_cdf(x, 3.5, 1) == pytest.approx(x ** 3.5, abs=1e-12)
    assert stats.beta_cdf(x, 1, 2.5) == pytest.approx(1 - (1 - x) ** 2.5, abs=1e-12)
    assert stats.beta_cdf(x, 2, 2) == pytest.approx(3 * x ** 2 - 2 * x ** 3, abs=1e-12)
    # Jeffreys prior Beta(0.5, 0.5) is the arcsine distribution
    assert stats.beta_cdf(x, 0.5, 0.5) == pytest.approx(2 / math.pi * math.asin(math.sqrt(x)), abs=1e-12)


@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.95])
def test_beta_ppf_inverts_closed_forms(q):
    assert stats.beta_ppf(q, 0.5, 0.5) == pytest.approx(math.sin(math.pi * q / 2) ** 2, abs=1e-12)
    assert stats.beta_ppf(q, 4, 1) == pytest.approx(q ** 0.25, abs=1e-12)
    assert stats.beta_ppf(q, 1, 3) == pytest.approx(1 - (1 - q) ** (1 / 3), abs=1e-12)


def test_beta_cdf_is_symmetric():
    for a, b, x in [(15.5, 1.5, 0.8), (0.5, 73.5, 0.03), (142.5, 4.5, 0.97)]:
        assert stats.beta_cdf(x, a, b) == pytest.approx(1 - stats.beta_cdf(1 - x, b, a), abs=1e-12)


def test_jeffreys_posterior():
    p = stats.posterior(15, 16)
    assert (p["a"], p["b"]) == (15.5, 1.5)
    assert p["mean"] == pytest.approx(15.5 / 17)
    assert stats.beta_cdf(p["lo"], 15.5, 1.5) == pytest.approx(0.05, abs=1e-9)
    assert stats.beta_cdf(p["hi"], 15.5, 1.5) == pytest.approx(0.95, abs=1e-9)
    empty = stats.posterior(0, 0)  # the prior itself: arcsine quantiles
    assert empty["lo"] == pytest.approx(math.sin(math.pi * 0.05 / 2) ** 2, abs=1e-12)
    assert empty["hi"] == pytest.approx(math.sin(math.pi * 0.95 / 2) ** 2, abs=1e-12)


def test_projected_informative_closed_forms():
    # p ~ Uniform: E[p^2] = E[(1-p)^2] = 1/3
    assert stats.projected_informative(1, 1, 2) == pytest.approx(1 / 3)
    # a near point mass at p=0.3
    assert stats.projected_informative(3e6, 7e6, 5) == pytest.approx(1 - 0.3 ** 5 - 0.7 ** 5, abs=1e-6)


# ---------------------------------------------------------------- grouping and excluded rows

def test_budget_rows_are_excluded_and_reported_apart():
    # review r5 H: rows that ran out of OUR calibration budget used to count as failures
    good = [row("qe_bugfix-d2", s, a, 1.0) for s in range(4) for a in range(2)]
    budget = [row("qe_bugfix-d2", s, a, 0.0, end=end) for s, end in
              zip(range(4, 7), ("global_budget", "context_budget", "completion_budget")) for a in range(2)]
    s = stats.summarize(good + budget)
    assert (s["budget"], s["errors"], s["valid"]) == (6, 0, 8)
    assert (s["solve"]["k"], s["solve"]["n"]) == (8, 8) and s["mean_reward"] == 1.0
    assert s["ends"]["global_budget"] == 2


def budget_row(end, seed, **limits):
    return row("qe_bugfix-d3", seed, 0, 0.0, end=end, **limits)


TEMPLATE_LIMITS = {"completion_tokens": 6144, "context_tokens": 12288}


@pytest.mark.parametrize("r, status", [
    (budget_row("global_budget", 1), "harness_budget"),  # the whole calibration run ran dry: never the model's
    (budget_row("context_budget", 1), "under_limit"),  # calibrate.py's default 8192 < the template's 12288
    (budget_row("context_budget", 1, episode_context_tokens=12288), "budget_failure"),
    (budget_row("completion_budget", 1, episode_completion_tokens=4096), "under_limit"),
    (budget_row("completion_budget", 1, episode_completion_tokens=8192), "budget_failure"),
    (row("qe_bugfix-d3", 1, 0, None), "error"),
    (row("qe_bugfix-d3", 1, 0, 0.0, end="step_cap"), "valid"),
])
def test_episode_status_compares_the_limits_an_episode_ran_under(r, status):
    assert stats.episode_status(r, TEMPLATE_LIMITS) == status
    assert stats.is_excluded(r, TEMPLATE_LIMITS) == (status in ("error", "harness_budget", "under_limit"))


def test_budget_ends_at_the_template_limits_are_failures(capsys):
    # review v2 c_budget: calibrated AT the template's limits, half the episodes ran out of context;
    # the arena would see the same, so the cell is mixed, not always solved
    rows = [row("qe_bugfix-d3", s, 0, 1.0) for s in range(16)]
    at_limit = rows + [budget_row("context_budget", 100 + s, episode_context_tokens=12288) for s in range(16)]
    cell = stats.compute_stats(at_limit, template())["cells"]["qe_bugfix-d3"]
    assert (cell["valid"], cell["budget_failures"], cell["budget"]) == (32, 16, 0)
    assert cell["projected_mixed"]["4"] == pytest.approx(stats.projected_mixed([1.0] * 16 + [0.0] * 16, 4))
    under = rows + [budget_row("context_budget", 100 + s) for s in range(16)]  # calibrate.py defaults
    cell = stats.compute_stats(under, template())["cells"]["qe_bugfix-d3"]
    assert (cell["valid"], cell["under_limit"]) == (16, 16)
    assert "ran under the submitted limits" in capsys.readouterr().err


def test_step_cap_is_a_real_failure():
    s = stats.summarize([row("qe_bugfix-d2", 1, 0, 0.0, end="step_cap"), row("qe_bugfix-d2", 1, 1, 1.0)])
    assert s["valid"] == 2 and s["budget"] == 0  # the arena caps steps too


def test_duplicate_keys_across_files_are_a_hard_error(tmp_path):
    # review r5 J: two runs that share seeds merged into fake mixed groups
    a = write_jsonl(tmp_path / "a.jsonl", [row("qe_bugfix-d2", 500000, 0, 1.0)])
    b = write_jsonl(tmp_path / "b.jsonl", [row("qe_bugfix-d2", 500000, 0, 0.0)])
    with pytest.raises(SystemExit, match="duplicate"):
        stats.load_runs([a, b])


def test_projected_mixed_counts_partial_credit():
    # binary rewards reduce exactly to the Jeffreys solve posterior
    assert stats.projected_mixed([1.0, 0.0, 0.0], 4) == pytest.approx(stats.projected_informative(1.5, 2.5, 4))
    # review r5 G: [2/3, 0, 0.5, 0] never solves, yet its groups are mixed
    partial = stats.projected_mixed([2 / 3, 0.0, 0.5, 0.0] * 8, 4)
    assert partial > 0.6 and stats.projected_informative(0.5, 32.5, 4) < 0.07
    # Dirichlet moments, by hand: values {0, 1} plus {0.5}; alphas 2.5, 0.5, 1.5 (total 4.5), g = 2
    expected = 1 - sum(a * (a + 1) / (4.5 * 5.5) for a in (2.5, 0.5, 1.5))
    assert stats.projected_mixed([0.0, 0.0, 0.5], 2) == pytest.approx(expected)



def test_groups_are_task_seed_corpus():
    rows = [row("qe_bugfix-d2", 1, 0, 1.0, "f"), row("qe_bugfix-d2", 1, 1, 1.0, "f"),
            row("qe_bugfix-d2", 1, 0, 0.0, "g"), row("qe_bugfix-d2", 1, 1, 0.0, "g")]
    s = stats.summarize(rows)
    assert s["groups"] == 2 and s["informative"]["k"] == 0  # pooled over corpus it would be 1/1
    pooled = stats.summarize(rows, by_corpus=False)
    assert pooled["groups"] == 1 and pooled["informative"]["k"] == 1


def test_groups_split_by_seed_and_task():
    rows = [row("qe_bugfix-d2", 1, 0, 1.0), row("qe_bugfix-d2", 2, 0, 0.0),
            row("qe_bugfix-d2", 1, 1, 1.0), row("qe_bugfix-d2", 2, 1, 0.0),
            row("qe_bugfix-d3", 1, 0, 0.0), row("qe_bugfix-d3", 1, 1, 0.5)]
    cell_stats = stats.compute_stats(rows)["cells"]
    assert cell_stats["qe_bugfix-d2"]["groups"] == 2 and cell_stats["qe_bugfix-d2"]["informative"]["k"] == 0
    assert cell_stats["qe_bugfix-d3"]["groups"] == 1 and cell_stats["qe_bugfix-d3"]["informative"]["k"] == 1


def test_equal_partial_rewards_are_not_informative():
    s = stats.summarize([row("qe_mutant_kill-d3", 1, 0, 2 / 3), row("qe_mutant_kill-d3", 1, 1, 0.6666666666666666)])
    assert s["informative"]["k"] == 0 and s["solve"]["k"] == 0


def test_error_rows_are_excluded_not_zeroed():
    rows = [row("qe_bugfix-d2", 1, 0, 1.0), row("qe_bugfix-d2", 1, 1, None, error="HTTP 500"),
            row("qe_bugfix-d2", 2, 0, 1.0), row("qe_bugfix-d2", 2, 1, 1.0)]
    s = stats.summarize(rows)
    assert (s["episodes"], s["errors"], s["valid"]) == (4, 1, 3)
    assert s["mean_reward"] == 1.0  # zeroing would give 0.75
    assert (s["solve"]["k"], s["solve"]["n"]) == (3, 3)
    assert (s["groups"], s["short_groups"]) == (1, 1)  # seed 1 has one valid episode left
    assert s["informative"]["k"] == 0  # zeroing would make seed 1 look informative
    assert s["ends"] == {"error": 1, "submit": 3}


def test_real_fixture_counts():
    rows = stats.load_runs(sorted(FIXTURES.glob("*.jsonl")))
    assert all("messages" not in r for r in rows)
    cell_stats = stats.compute_stats(rows)
    kill = cell_stats["cells"]["qe_mutant_kill-d1"]  # seeds 830000-3: [1,.5] [0,0] [1,1] [1,1]
    assert (kill["solve"]["k"], kill["solve"]["n"], kill["informative"]["k"], kill["groups"]) == (5, 8, 1, 4)
    dead = cell_stats["cell_corpus"]["qe_bugfix-d2|alternate_chat_roles"]
    assert (dead["errors"], dead["valid"], dead["groups"]) == (4, 0, 0)
    split = cell_stats["cell_corpus"]["qe_bugfix-d2|flaky_run_summary"]  # 2/4 solved, but per seed 2/2 and 0/2
    assert split["solve"]["k"] == 2 and split["informative"]["k"] == 0


# ---------------------------------------------------------------- select

def synthetic_stats(cells: dict[str, list[tuple[int, list[float]]]]) -> dict:
    rows = [row(t, seed, a, r) for t, groups in cells.items() for seed, rewards in groups
            for a, r in enumerate(rewards)]
    return stats.compute_stats(rows)


def template():
    return json.loads(TEMPLATE.read_text())


MIXED = {
    "qe_mutant_kill-d1": [(s, [1.0, 0.0]) for s in range(4)],
    "qe_mutant_kill-d2": [(s, [1.0, 1.0]) for s in range(8)],  # always solved
    "qe_mutant_kill-d3": [(s, [0.0, 0.0]) for s in range(8)],  # never solved
    "qe_bugfix-d2": [(0, [1.0, 0.0]), (1, [1.0, 1.0]), (2, [0.0, 0.0])],
    "qe_bugfix-d3": [(s, [1.0, 1.0]) for s in range(7)] + [(7, [1.0, 0.0])],
}


def test_select_excludes_rarely_mixed_cells_on_both_sides():
    tasks, reasons = sel.select_cells(synthetic_stats(MIXED), template(), max_cells=6, seed=1)
    by_id = {r["task_id"]: r for r in reasons}
    assert {t["task_id"] for t in tasks} == {"qe_mutant_kill-d1", "qe_bugfix-d2", "qe_bugfix-d3"}
    assert "rarely mixed (mostly high reward)" in by_id["qe_mutant_kill-d2"]["reason"]  # 16/16: 0.10
    assert "rarely mixed (mostly low reward)" in by_id["qe_mutant_kill-d3"]["reason"]
    assert by_id["qe_bugfix-d3"]["p_mixed"] == pytest.approx(stats.projected_informative(15.5, 1.5, 4))
    strict, _ = sel.select_cells(synthetic_stats(MIXED), template(), 6, 1, min_mixed=0.3)
    assert "qe_bugfix-d3" not in {t["task_id"] for t in strict}  # 15/16 has p_mix@4 0.29


def test_dead_threshold_is_posterior_mean_based():
    # review r5 I: 0/8 stays eligible as exploration (p_mix@4 0.18); 0/16 is excluded (0.10)
    eight = synthetic_stats({"qe_mutant_kill-d3": [(s, [0.0, 0.0]) for s in range(4)]})
    sixteen = synthetic_stats({"qe_mutant_kill-d3": [(s, [0.0, 0.0]) for s in range(8)]})
    assert sel.select_cells(eight, template(), 6, 1)[0]
    assert not sel.select_cells(sixteen, template(), 6, 1)[0]


def test_select_takes_partial_credit_cells():
    # review r5 G: a mutant-kill cell that pays 0.5 for a distinguishing input is what we want
    cells = synthetic_stats({"qe_mutant_kill-d3": [(s, [2 / 3, 0.0, 0.5, 0.0]) for s in range(8)]})
    tasks, reasons = sel.select_cells(cells, template(), 6, 1)
    assert [t["task_id"] for t in tasks] == ["qe_mutant_kill-d3"] and reasons[-1]["solve"]["k"] == 0


def test_select_on_one_attempt_per_seed_data(tmp_path):
    # review r4 F: DESIGN's own --attempts 1 calibration has no same-seed groups at all
    ids = [e["task_id"] for e in template()]
    runs = write_jsonl(tmp_path / "att1.jsonl", [row(t, 900000 + s, 0, float(s % 2)) for t in ids for s in range(16)])
    tasks, _ = sel.select_cells(stats.compute_stats(stats.load_runs([runs])), template(), 6, 1)
    assert len(tasks) == 5
    _, reasons = sel.select_cells(stats.compute_stats(stats.load_runs([runs])), template(), 6, 1, objective="group")
    assert all("same-seed groups" in r["reason"] for r in reasons)


def test_select_cli_exits_nonzero_when_nothing_qualifies(tmp_path):
    runs = write_jsonl(tmp_path / "r.jsonl", [row("qe_bugfix-d2", s, 0, 1.0) for s in range(20)])
    code = cli.main(["select", str(runs), "--max-cells", "6", "--seed", "1", "--template", str(TEMPLATE),
                     "--out", str(tmp_path / "t.json")])
    assert code == 4 and not (tmp_path / "t.json").exists()  # nothing to submit, so no task list
    assert (tmp_path / "t.why.json").exists()


def test_select_is_deterministic_and_respects_max_cells():
    cell_stats = synthetic_stats({e["task_id"]: [(s, [1.0, 0.0]) for s in range(3)] for e in template()})
    first = sel.select_cells(cell_stats, template(), max_cells=2, seed=42)
    assert first == sel.select_cells(cell_stats, template(), max_cells=2, seed=42)
    assert len(first[0]) == 2
    assert sum(r["decision"] == "selected" for r in first[1]) == 2


def test_select_preserves_every_template_field():
    tmpl = template()
    original = json.loads(json.dumps(tmpl))
    tasks, _ = sel.select_cells(synthetic_stats(MIXED), tmpl, max_cells=6, seed=3)
    by_id = {e["task_id"]: e for e in original}
    for task in tasks:
        assert task == by_id[task["task_id"]]
    tasks[0]["completion_tokens"] = 1
    assert tmpl == original  # output is a copy


def test_select_records_cells_without_a_template_entry():
    cell_stats = synthetic_stats({"qe_bugfix-d1": [(s, [1.0, 0.0]) for s in range(3)]})
    tasks, reasons = sel.select_cells(cell_stats, template(), 6, 1)
    by_id = {r["task_id"]: r for r in reasons}
    assert not tasks
    assert "no template entry" in by_id["qe_bugfix-d1"]["reason"]
    assert by_id["qe_mutant_kill-d1"]["reason"] == "no calibration data"


def test_select_cli_writes_identical_files_and_sidecar(tmp_path):
    runs = sorted(str(p) for p in FIXTURES.glob("*.jsonl"))
    outs = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.json"
        assert cli.main(["select", *runs, "--max-cells", "6", "--seed", "7", "--template", str(TEMPLATE),
                         "--out", str(out)]) == 0
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()
    why = json.loads((tmp_path / "a.why.json").read_text())
    assert why["seed"] == 7 and len(why["runs"]) == 2 and all(c.get("reason") for c in why["cells"])
    assert why["objective"] == "projected" and "p_mix@G" in why["measure"]


SEED_SPLIT = {  # A: solved on half the seeds, never mixed within a seed; B: mostly solved, mixed twice
    "qe_mutant_kill-d1": [(s, [1.0, 1.0]) for s in range(4)] + [(s, [0.0, 0.0]) for s in range(4, 8)],
    "qe_mutant_kill-d2": [(s, [1.0, 0.0]) for s in range(2)] + [(s, [1.0, 1.0]) for s in range(2, 8)],
}


def winners(objective, n=200):
    cell_stats = synthetic_stats(SEED_SPLIT)
    return Counter(sel.select_cells(cell_stats, template(), 1, seed, objective=objective)[0][0]["task_id"]
                   for seed in range(n))


def test_select_defaults_to_the_projected_trainer_group():
    _, reasons = sel.select_cells(synthetic_stats(MIXED), template(), 6, 1)
    assert all("p_mix@4" in r["reason"] for r in reasons if r["decision"] == "selected")
    _, reasons = sel.select_cells(synthetic_stats(MIXED), template(), 6, 1, objective="group")
    assert all("same-seed" in r["reason"] for r in reasons if r["decision"] == "selected")
    with pytest.raises(ValueError):
        sel.select_cells(synthetic_stats(MIXED), template(), 6, 1, objective="other")


def test_projected_objective_credits_cells_that_mix_across_instances():
    # Same-seed groups call A uninformative, but 4 fresh instances of a p~0.5 cell are mixed 87% of
    # the time, against about 48% for B at p~0.85. The two objectives must disagree on this data.
    assert winners("projected")["qe_mutant_kill-d1"] > 160
    assert winners("group")["qe_mutant_kill-d2"] > 160


def test_stats_reports_both_projected_group_sizes():
    s = stats.summarize([row("qe_bugfix-d2", 1, 0, 1.0), row("qe_bugfix-d2", 1, 1, 0.0)])
    a, b = s["solve"]["a"], s["solve"]["b"]
    assert s["projected_mixed"] == {g: pytest.approx(stats.projected_informative(a, b, int(g))) for g in ("4", "5")}


def test_modules_stay_under_500_lines():
    for path in [ROOT / "scripts" / "flywheel.py", *(ROOT / "scripts" / "qe_flywheel").glob("*.py")]:
        assert len(path.read_text().splitlines()) < 500, path


def test_cells_with_too_few_valid_episodes_are_excluded():
    cell_stats = synthetic_stats({"qe_mutant_kill-d1": [(0, [1.0, 0.0, 1.0])]})  # mixed, but 3 episodes
    tasks, reasons = sel.select_cells(cell_stats, template(), 6, 1)
    assert not tasks and "3 valid episodes < 4" in next(r for r in reasons if r["task_id"] == "qe_mutant_kill-d1")["reason"]
    assert sel.select_cells(cell_stats, template(), 6, 1, min_episodes=3)[0]
