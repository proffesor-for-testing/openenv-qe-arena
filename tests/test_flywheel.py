"""Flywheel tooling: posterior math, grouping, selection, the ledger's null, and one-knob decisions."""

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
from qe_flywheel import ledger as lg  # noqa: E402
from qe_flywheel import search, stats  # noqa: E402
from qe_flywheel import select as sel  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "flywheel"
TEMPLATE = ROOT / "submission" / "tasks-v1.json"
RECEIPT = ROOT / "submission" / "receipt-v1.json"


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


# ---------------------------------------------------------------- grouping and error rows

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


def test_select_excludes_saturated_and_dead():
    tasks, reasons = sel.select_cells(synthetic_stats(MIXED), template(), max_cells=6, seed=1)
    by_id = {r["task_id"]: r for r in reasons}
    assert {t["task_id"] for t in tasks} == {"qe_mutant_kill-d1", "qe_bugfix-d2"}
    assert by_id["qe_mutant_kill-d2"]["reason"].startswith("saturated")
    assert by_id["qe_mutant_kill-d3"]["reason"].startswith("dead")
    assert by_id["qe_bugfix-d3"]["reason"].startswith("saturated")  # 15/16 -> 0.912
    loose, _ = sel.select_cells(synthetic_stats(MIXED), template(), 6, 1, saturated=0.95)
    assert "qe_bugfix-d3" in {t["task_id"] for t in loose}


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
    cell_stats = synthetic_stats({"qe_bugfix-d1": [(0, [1.0, 0.0])]})
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
    assert all("p_inf@4" in r["reason"] for r in reasons if r["decision"] == "selected")
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
    assert s["projected_informative"] == {g: pytest.approx(stats.projected_informative(a, b, int(g)))
                                          for g in ("4", "5")}


def test_modules_stay_under_500_lines():
    for path in [ROOT / "scripts" / "flywheel.py", *(ROOT / "scripts" / "qe_flywheel").glob("*.py")]:
        assert len(path.read_text().splitlines()) < 500, path


# ---------------------------------------------------------------- ledger

BOARD = FIXTURES / "leaderboard-trim.json"  # trimmed real GET /api/leaderboard response


def board():
    return json.loads(BOARD.read_text())


def run_record(run_id, submission_id, industrial, office, user="Profa-QE"):
    """A real leaderboard runs[] record with its identity and two domain scores rewritten."""
    run = json.loads(json.dumps(board()["runs"][0]))
    for d in run["scores"].values():
        d.update(solved=0, score=0.0)
    run["scores"]["industrial-physical-systems"].update(solved=industrial, score=industrial / 5)
    run["scores"]["office-white-collar"].update(solved=office, score=office / 5)
    run.update(run_id=run_id, submission_id=submission_id, user=user, solved=industrial + office)
    return run


def write_receipt(directory, version, submission_id):
    request = json.loads((ROOT / "submission" / "request-v1.json").read_text())
    request["submission_id"] = submission_id
    (directory / f"request-{version}.json").write_text(json.dumps(request))
    receipt = {"submission_id": submission_id, "request_sha256": lg.canonical_sha(request)}
    (directory / f"receipt-{version}.json").write_text(json.dumps(receipt))
    return directory / f"receipt-{version}.json"


def add(ledger, tmp_path, receipt, run):
    path = tmp_path / f"{run['run_id']}.json"
    path.write_text(json.dumps(run))
    return lg.ledger_add(ledger, receipt, path)


def test_parse_result_reads_real_leaderboard_records():
    for run in board()["runs"]:
        run_id, domains = lg.parse_result(run)
        assert run_id == run["run_id"] and set(domains) == set(lg.DOMAIN_IDS)
        assert sum(d["solved"] for d in domains.values()) == run["solved"]
        assert all(d["total"] == 5 for d in domains.values())


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.update(solved=r["solved"] + 1), "disagrees"),
    (lambda r: r.pop("submission_id"), "missing"),
    (lambda r: r["scores"].update(sports={"solved": 0, "total": 5}), "unknown domain"),
    (lambda r: r["scores"]["cybersecurity"].update(solved=6), "solved <= total"),
])
def test_parse_result_rejects_malformed_records(mutate, message):
    run = board()["runs"][0]
    mutate(run)
    with pytest.raises(SystemExit, match=message):
        lg.parse_result(run)


def test_ledger_is_append_only(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    first = add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", 2, 1))
    assert first["cells"][0] == "qe_mutant_kill-d1" and first["request_sha256"].startswith("e6a38411")
    before = ledger.read_bytes()
    add(ledger, tmp_path, write_receipt(tmp_path, "v2", "profa-qe-arena-v2"),
        run_record("run-2", "profa-qe-arena-v2", 0, 2))
    after = ledger.read_bytes()
    assert after.startswith(before) and after.count(b"\n") == 2
    with pytest.raises(SystemExit, match="never replaced"):
        add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", 2, 2))
    assert ledger.read_bytes() == after
    assert [e["seq"] for e in lg.read_ledger(ledger)] == [1, 2]
    ledger.write_bytes(after.replace(b'"solved":3', b'"solved":4', 1))
    with pytest.raises(SystemExit, match="chain broken"):
        lg.read_ledger(ledger)


def test_ledger_refuses_a_run_for_another_submission(tmp_path):
    with pytest.raises(SystemExit, match="run is for"):
        add(tmp_path / "l.jsonl", tmp_path, RECEIPT, board()["runs"][0])


def test_ledger_refuses_a_request_that_does_not_match_the_receipt(tmp_path):
    request = json.loads((ROOT / "submission" / "request-v1.json").read_text())
    request["tasks"] = request["tasks"][:1]
    (tmp_path / "request-v1.json").write_text(json.dumps(request))
    (tmp_path / "receipt-v1.json").write_text(RECEIPT.read_text())
    with pytest.raises(SystemExit, match="hashes to"):
        add(tmp_path / "l.jsonl", tmp_path, tmp_path / "receipt-v1.json",
            run_record("x", "profa-qe-arena-v1", 1, 1))


def test_fetch_appends_matching_runs_once_and_saves_the_baseline(tmp_path):
    lb = board()
    ruv = [r for r in lb["runs"] if r["user"] == "ruv"]
    for i, run in enumerate(ruv):
        write_receipt(tmp_path, f"v{i}", run["submission_id"])
    ledger = tmp_path / "ledger.jsonl"
    calls = []
    get = lambda url: calls.append(url) or lb  # noqa: E731 - the network, replaced by the saved response
    first = lg.ledger_fetch(ledger, "ruv", receipts=tmp_path, get=get)
    assert calls == [lg.LEADERBOARD]
    assert first["matched"] == 2 and sorted(first["appended"]) == sorted(r["run_id"] for r in ruv)
    entries = lg.read_ledger(ledger)
    assert all(e["user"] == "ruv" and e["total"] == 40 for e in entries)
    before = ledger.read_bytes()
    again = lg.ledger_fetch(ledger, "ruv", receipts=tmp_path, board=lb)
    assert again["appended"] == [] and len(again["already"]) == 2 and ledger.read_bytes() == before
    baseline, source = lg.resolve_baseline(ledger, None)
    assert baseline == {"industrial-physical-systems": 2, "office-white-collar": 2}
    assert "leaderboard" in source


def test_fetch_filters_by_submission_and_skips_runs_without_a_receipt(tmp_path):
    lb = board()
    target = next(r for r in lb["runs"] if r["user"] == "ruv")
    ledger = tmp_path / "new-dir" / "l.jsonl"  # a fresh directory: nothing appended, baseline still saved
    report = lg.ledger_fetch(ledger, "ruv", target["submission_id"], tmp_path, board=lb)
    assert report["matched"] == 1 and report["appended"] == []
    assert report["no_receipt"] == [f"{target['submission_id']}/{target['run_id']}"]
    assert not ledger.exists() and lg.baseline_path(ledger).exists()
    assert lg.ledger_fetch(ledger, "Profa-QE", board=lb, receipts=tmp_path)["matched"] == 0


def test_fetch_follows_and_validates_cursors():
    pages = {lg.LEADERBOARD: dict(board(), next_cursor="12-34"),
             lg.LEADERBOARD + "/12-34": {"runs": [{"run_id": "extra"}], "next_cursor": None}}
    merged = lg.fetch_leaderboard(pages.__getitem__)
    assert merged["runs"][-1] == {"run_id": "extra"} and len(merged["runs"]) == 4
    calls = []
    with pytest.raises(SystemExit, match="cursor"):
        lg.fetch_leaderboard(lambda url: calls.append(url) or dict(board(), next_cursor="../admin"))
    assert calls == [lg.LEADERBOARD]  # the bad cursor is never requested


def test_offline_fallback_baseline_matches_the_leaderboard(tmp_path):
    baseline, source = lg.resolve_baseline(tmp_path / "ledger.jsonl", None)
    assert "OFFLINE FALLBACK" in source
    assert baseline == lg.baseline_from(board()) == lg.parse_baseline(str(BOARD))


def test_composite_null_hand_worked():
    # Two domains of 2 tasks, baseline p = 1/2, k = 2 runs. Bin(2, .5) has F(0)=1/4, F(1)=3/4, so
    # E[max] = (1 - (1/4)^2) + (1 - (3/4)^2) = 15/16 + 7/16 = 11/8 per domain; 11/4 for the composite.
    dist = lg.null_composite(2, {"industrial": (2, 0.5), "office": (2, 0.5)})
    assert float(np.dot(np.arange(len(dist)), dist)) == pytest.approx(2.75)
    assert lg.tail(dist, 4) == pytest.approx((7 / 16) ** 2)  # both maxes hit 2: P(max=2) = 1 - (3/4)^2


def test_null_from_the_real_leaderboard_baseline():
    entries = [{"domains": lg.parse_result(r)[1]} for r in board()["runs"]]
    nulls = lg.null_domains(entries, lg.baseline_from(board()))
    assert nulls["industrial-physical-systems"] == (5, 0.4) and nulls["software-engineering"] == (5, 0.0)
    one = lg.null_composite(1, nulls)  # one run of the baseline scores the baseline
    assert float(np.dot(np.arange(len(one)), one)) == pytest.approx(4.0)
    three = lg.null_composite(3, nulls)  # 2 * E[max of 3 Bin(5, .4)], computed independently
    cdf = np.cumsum([math.comb(5, x) * 0.4 ** x * 0.6 ** (5 - x) for x in range(6)])
    assert float(np.dot(np.arange(len(three)), three)) == pytest.approx(
        2 * sum(1 - cdf[x - 1] ** 3 for x in range(1, 6)))


def test_ledger_show_labels_composite_against_null(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", 2, 0))
    add(ledger, tmp_path, write_receipt(tmp_path, "v2", "profa-qe-arena-v2"),
        run_record("run-2", "profa-qe-arena-v2", 0, 2))
    summary = lg.ledger_show(lg.read_ledger(ledger), lg.parse_baseline(lg.BASELINE), source="test")
    printed = capsys.readouterr().out
    assert summary["composite"] == 4 and summary["best"]["office-white-collar"] == 2
    two = lg.null_composite(2, lg.null_domains(lg.read_ledger(ledger), lg.parse_baseline(lg.BASELINE)))
    assert summary["null_expected"] == pytest.approx(float(np.dot(np.arange(len(two)), two)))
    assert summary["null_p_value"] == pytest.approx(lg.tail(two, 4))
    assert "PER-RUN SCORES" in printed and "selection-inflated" in printed and "baseline source: test" in printed


# ---------------------------------------------------------------- onecell-search

def config():
    return search.init_config(template())


def test_init_config_reads_environment_and_template():
    c = config()
    assert c["max_steps"] == 12 and c["max_test_runs"] == 3
    assert c["difficulty_mix"] == {"d1": 0.2, "d2": 0.4, "d3": 0.4}
    assert set(c["completion_tokens"]) == {e["task_id"] for e in template()}


@pytest.mark.parametrize("seed", range(40))
def test_propose_changes_exactly_one_knob(seed):
    seen = set(range(830_000, 830_008))
    plan = search.propose(config(), seed, seen, [], sorted(config()["completion_tokens"]))
    assert search.changed_knobs(plan["baseline"], plan["candidate"]) == [plan["knob"]]
    assert plan["from"] != plan["to"]
    disc, conf = set(plan["discovery"]["seeds"]), set(plan["confirmation"]["seeds"])
    assert not disc & conf and not conf & seen and not disc & seen
    assert plan == search.propose(config(), seed, seen, [], sorted(config()["completion_tokens"]))


def test_mix_change_counts_as_one_knob():
    c = config()
    moved = search.apply_move(c, "difficulty_mix", {"d1": 0.1, "d2": 0.5, "d3": 0.4})
    assert search.changed_knobs(c, moved) == ["difficulty_mix"]


def test_propose_skips_tried_moves():
    c = config()
    history = [{"knob": k, "to": v} for k, v in search.candidate_moves(c)[1:]]
    plan = search.propose(c, 0, set(), history, ["qe_bugfix-d2"])
    assert (plan["knob"], plan["to"]) == search.candidate_moves(c)[0]


def split_rows(split, reward_of):
    return [row(t, s, a, reward_of(t, s, a)) for t in split["task_ids"] for s in split["seeds"]
            for a in range(split["attempts"])]


def make_plan():
    c = config()
    c["difficulty_mix"] = {"d1": 0.0, "d2": 1.0, "d3": 0.0}
    plan = search.propose(c, 5, set(), [], ["qe_bugfix-d2"], discovery_seeds=4, confirm_seeds=4, attempts=2)
    if plan["knob"] == "difficulty_mix":  # keep the reweighting out of this test
        plan = search.propose(c, 6, set(), [], ["qe_bugfix-d2"], discovery_seeds=4, confirm_seeds=4, attempts=2)
    assert plan["knob"] != "difficulty_mix"
    return plan


def decide_with(plan, base_d, cand_d, base_c, cand_c):
    return search.decide(plan, split_rows(plan["discovery"], base_d), split_rows(plan["discovery"], cand_d),
                     split_rows(plan["confirmation"], base_c), split_rows(plan["confirmation"], cand_c))


def test_decide_rejects_confirmation_loss_despite_discovery_gain():
    plan = make_plan()
    half = lambda t, s, a: float(a == 0)  # noqa: E731 - 0.5 mean
    full = lambda t, s, a: 1.0  # noqa: E731
    worse = lambda t, s, a: 0.0 if s == plan["confirmation"]["seeds"][0] else float(a == 0)  # noqa: E731
    result = decide_with(plan, half, full, half, worse)
    assert result["discovery"]["delta"] == pytest.approx(0.5)
    assert result["confirmation"]["delta"] < 0
    assert result["accept"] is False and any("confirmation" in r for r in result["reasons"])


def test_decide_accepts_gain_with_no_confirmation_loss_and_respects_margin():
    plan = make_plan()
    half = lambda t, s, a: float(a == 0)  # noqa: E731
    full = lambda t, s, a: 1.0  # noqa: E731
    assert decide_with(plan, half, full, half, half)["accept"] is True
    tiny = lambda t, s, a: 1.0 if (a == 1 and s == plan["discovery"]["seeds"][0]) else float(a == 0)  # noqa: E731
    small = decide_with(plan, half, tiny, half, half)  # +1/8 = 0.125 > 0.05 margin
    assert small["accept"] is True
    plan["rule"]["margin"] = 0.2
    assert decide_with(plan, half, tiny, half, half)["accept"] is False


def test_decide_refuses_unpaired_or_foreign_rows():
    plan = make_plan()
    half = lambda t, s, a: float(a == 0)  # noqa: E731
    base = split_rows(plan["discovery"], half)
    with pytest.raises(SystemExit, match="unpaired"):
        search.decide(plan, base, base[:-1], split_rows(plan["confirmation"], half),
                  split_rows(plan["confirmation"], half))
    with pytest.raises(SystemExit, match="outside the plan"):
        search.decide(plan, base, base, base, base)  # discovery rows used as confirmation


def test_decide_drops_error_pairs():
    plan = make_plan()
    half = lambda t, s, a: float(a == 0)  # noqa: E731
    seed0 = plan["discovery"]["seeds"][0]
    errs = lambda t, s, a: None if (s == seed0 and a == 1) else float(a == 0)  # noqa: E731
    result = decide_with(plan, half, errs, half, half)
    assert result["discovery"]["dropped_error_pairs"] == 1
    assert result["discovery"]["delta"] == pytest.approx(4 / 7 - 4 / 7)
