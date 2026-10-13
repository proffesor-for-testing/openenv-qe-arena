"""Domain cells: task IDs scoped to an arena domain draw only that domain's corpus, and the frozen
software-engineering cells are unchanged (docs/DESIGN.md, "Domain cells")."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_dataset import domain_label  # noqa: E402
from oracle import solve_actions  # noqa: E402
from qe_arena import corpus, tasks  # noqa: E402
from qe_arena.domains import BY_SLUG, DOMAINS, SLUGS, SOFTWARE  # noqa: E402
from qe_arena.environment import QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import (ALL_TASK_IDS, DOMAIN_TASK_IDS, FAMILIES, LEVELS, MIN_DOMAIN_ENTRIES,  # noqa: E402
                            TASK_IDS, format_task_id, make_task, parse_task_id, task_domain)
from qe_flywheel import select as sel  # noqa: E402
from qe_flywheel import stats  # noqa: E402

FROZEN = ["qe_bugfix-d1", "qe_bugfix-d2", "qe_bugfix-d3", "qe_mutant_kill-d1", "qe_mutant_kill-d2",
          "qe_mutant_kill-d3", "qe_flaky-d1", "qe_flaky-d2", "qe_flaky-d3", "qe_bugfix-d4"]


def test_slugs_name_every_domain_but_software_once():
    assert DOMAINS[0] == SOFTWARE and len(set(DOMAINS)) == 8
    assert set(SLUGS) == set(DOMAINS) - {SOFTWARE}
    assert len(BY_SLUG) == len(SLUGS) and all(slug.isalpha() and slug.islower() for slug in BY_SLUG)


def test_software_cells_keep_their_ids_order_and_pool():
    assert TASK_IDS == FROZEN and ALL_TASK_IDS[:len(FROZEN)] == FROZEN
    assert all(task_domain(t) == SOFTWARE for t in TASK_IDS)
    # The software pool is exactly what every unscoped ID drew before domains existed.
    software = [n for n, e in corpus.ENTRIES.items() if e.domain == SOFTWARE and not e.real_faults]
    assert corpus.domain_names(SOFTWARE) == tuple(software)
    assert set(corpus.REAL_NAMES) <= {n for n, e in corpus.ENTRIES.items() if e.domain == SOFTWARE}


def test_every_entry_names_a_known_domain():
    assert {e.domain for e in corpus.ENTRIES.values()} <= set(DOMAINS)
    assert sum(len(corpus.domain_names(d)) for d in DOMAINS) == len(corpus.INJECTED_NAMES)


def test_a_domain_is_served_exactly_when_its_pool_reaches_the_minimum():
    for domain in DOMAINS[1:]:
        size = len(corpus.domain_names(domain))
        for family in FAMILIES:
            ids = [f"{family}-{SLUGS[domain]}-d{level}" for level in LEVELS[family]]
            served = size >= MIN_DOMAIN_ENTRIES[family]
            assert all((t in DOMAIN_TASK_IDS) == served for t in ids), (domain, family, size)
            for t in ids:
                if not served:
                    with pytest.raises(ValueError):
                        parse_task_id(t)


def test_the_minimum_is_enforced_when_a_pool_shrinks(monkeypatch):
    finance = corpus.domain_names("finance-economics")
    assert len(finance) >= MIN_DOMAIN_ENTRIES["qe_mutant_kill"]
    real = corpus.domain_names

    def shrunk(domain, names=corpus.INJECTED_NAMES):
        found = real(domain, names)
        return found[:MIN_DOMAIN_ENTRIES["qe_mutant_kill"] - 1] if domain == "finance-economics" else found

    monkeypatch.setattr(corpus, "domain_names", shrunk)
    cells = tasks._domain_cells()
    assert not [t for t in cells if "-finance-" in t]
    assert [t for t in cells if t in TASK_IDS] == TASK_IDS


def test_domain_ids_round_trip_and_bad_ids_are_refused():
    assert {task_domain(t) for t in DOMAIN_TASK_IDS} == set(DOMAINS[1:]), "v4 serves every arena domain"
    for task_id in ALL_TASK_IDS:
        family, level = parse_task_id(task_id)
        assert format_task_id(family, level, task_domain(task_id)) == task_id
    assert format_task_id("qe_bugfix", 2) == "qe_bugfix-d2"
    for bad in ("qe_bugfix-software-d1", "qe_bugfix-finance-d5", "qe_mutant_kill-finance-d4", "qe_bugfix-bank-d1",
                "qe_bugfix--d1", "qe_bugfix-finance-economics-d1", "qe_bugfix-Finance-d1", "qe_bugfix-finance-d1 "):
        with pytest.raises(ValueError):
            parse_task_id(bad)
    with pytest.raises(ValueError):
        format_task_id("qe_bugfix", 1, "no-such-domain")


@pytest.mark.parametrize("task_id", [t for t in DOMAIN_TASK_IDS if not t.startswith("qe_flaky")])
def test_domain_cells_draw_only_their_domain(task_id):
    domain = task_domain(task_id)
    drawn = {make_task(task_id, seed).private["corpus"] for seed in range(2)}
    assert all(corpus.ENTRIES[n].domain == domain and not corpus.ENTRIES[n].real_faults for n in drawn)


def test_a_calibration_pool_must_match_the_cell_domain():
    finance = corpus.domain_names("finance-economics")[0]
    with pytest.raises(ValueError):
        make_task("qe_mutant_kill-d1", 0, pool=(finance,))
    with pytest.raises(ValueError):
        make_task("qe_mutant_kill-finance-d1", 0, pool=("backoff_schedule",))
    assert make_task("qe_mutant_kill-finance-d1", 0, pool=(finance,)).private["corpus"] == finance
    assert set(tasks.cell_names("qe_bugfix-finance-d1")) == set(corpus.domain_names("finance-economics"))


def _check_oracle_and_nothing(task_id: str, seed: int) -> None:
    task = make_task(task_id, seed)
    for actions, reward in ((solve_actions(task), 1.0), ([{"op": "read", "path": "*"},
                                                          {"op": "submit", "answer": {}}], 0.0)):
        env = QEEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        for action in actions:
            obs = env.step(QEAction(**action))
            if obs.done:
                break
        assert obs.done and obs.reward == reward, (task_id, seed, reward)


@pytest.mark.parametrize("task_id", [t for t in DOMAIN_TASK_IDS if not t.startswith("qe_flaky")])
def test_oracle_earns_one_and_doing_nothing_zero_on_a_domain_cell(task_id):
    _check_oracle_and_nothing(task_id, 20261013)


@pytest.mark.parametrize("task_id", [t for t in DOMAIN_TASK_IDS if t.startswith("qe_flaky")])
def test_flaky_oracle_earns_one_and_doing_nothing_zero_on_a_domain_cell(task_id):
    _check_oracle_and_nothing(task_id, 20261013)


@pytest.mark.parametrize("task_id", DOMAIN_TASK_IDS)
def test_oracle_earns_one_on_many_seeds_for_every_domain_cell(task_id):
    for seed in range(3_000_000, 3_000_008):
        _check_oracle_and_nothing(task_id, seed)


def test_task_listing_names_every_served_cell():
    listed = json.loads((ROOT / "tasks.json").read_text())["tasks"]
    assert [t["task_id"] for t in listed] == ALL_TASK_IDS
    assert {t["split"] for t in listed} == {"train"}


def test_exported_domain_label_keeps_the_software_rows_value():
    assert domain_label("qe_bugfix-d4") == "software engineering"  # what the frozen rows carry
    assert domain_label("qe_mutant_kill-finance-d2") == "finance economics"


def test_flywheel_reads_domains_from_task_ids():
    assert stats.cell_domain("qe_flaky-d3") == SOFTWARE
    assert stats.cell_domain("qe_flaky-industrial-d3") == "industrial-physical-systems"
    assert stats.cell_domain("qe_flaky-bank-d3") == "unknown" and stats.cell_domain("junk") == "unknown"
    assert stats.base_cell("qe_bugfix-finance-d4") == "qe_bugfix-d4" and stats.base_cell("qe_bugfix-d4") == "qe_bugfix-d4"
    for task_id in ALL_TASK_IDS:
        assert stats.cell_domain(task_id) == task_domain(task_id)


def test_flywheel_pools_domains_with_each_cells_own_limits():
    def row(task_id, seed, reward, end="submit"):
        return {"task_id": task_id, "seed": seed, "attempt": 0, "reward": reward, "end": end, "steps": 3, "invalid": 0}

    rows = [row("qe_mutant_kill-finance-d1", 1, 1.0), row("qe_mutant_kill-finance-d2", 2, 0.0),
            row("qe_mutant_kill-finance-d2", 3, 0.0, "context_budget"), row("qe_mutant_kill-d1", 4, 1.0)]
    template = [{"task_id": "qe_mutant_kill-finance-d2", "completion_tokens": 4096, "context_tokens": 8192}]
    out = stats.compute_stats(rows, template)
    finance = out["domains"]["finance-economics"]
    # The d2 budget end ran at its cell's limit, so it is a real failure and counts.
    assert finance["episodes"] == 3 and finance["valid"] == 3 and finance["solve"]["k"] == 1
    assert out["domains"][SOFTWARE]["valid"] == 1


def test_domain_template_copies_base_limits_and_skips_uncovered_cells():
    template = [{"task_id": "qe_mutant_kill-d2", "completion_tokens": 6144, "split": "train"}]
    cells = ["qe_mutant_kill-finance-d2", "qe_bugfix-finance-d1", "qe_mutant_kill-d2"]
    out, added, skipped = sel.extend_template(template, cells)
    assert added == ["qe_mutant_kill-finance-d2"] and skipped == ["qe_bugfix-finance-d1"]
    assert out[0] == template[0] and out[1] == {"task_id": "qe_mutant_kill-finance-d2", "completion_tokens": 6144,
                                                 "split": "train"}
    with pytest.raises(ValueError):
        sel.extend_template(template + template, cells)
