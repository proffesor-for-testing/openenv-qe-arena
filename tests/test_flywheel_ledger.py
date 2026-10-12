"""Flywheel ledger: leaderboard records, append-only integrity, fetch, and the composite's null."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flywheel as cli  # noqa: E402
from qe_flywheel import ledger as lg  # noqa: E402
from qe_flywheel import empirical, null, report  # noqa: E402

BOARD = Path(__file__).parent / "fixtures" / "flywheel" / "leaderboard-trim.json"  # trimmed real response
FULL = Path(__file__).parent / "fixtures" / "flywheel" / "leaderboard-2026-10-12.json"  # all 39 runs, after v1 scored
RECEIPT = ROOT / "submission" / "receipt-v1.json"
RECORDED = {"state": "recorded", "post_attempted": True}


def board():
    return json.loads(BOARD.read_text())


def run_record(run_id, submission_id, scores, user="Profa-QE", benchmark=None):
    """A real leaderboard runs[] record with identity and scores rewritten."""
    run = copy.deepcopy(board()["runs"][0])
    for name, d in run["scores"].items():
        d.update(solved=scores.get(name, 0), score=scores.get(name, 0) / 5)
    run.update(run_id=run_id, submission_id=submission_id, user=user, solved=sum(scores.values()))
    if benchmark:
        run["benchmark"] = benchmark
    return run


def write_receipt(directory, version, submission_id, **receipt_fields):
    request = json.loads((ROOT / "submission" / "request-v1.json").read_text())
    request["submission_id"] = submission_id
    (directory / f"request-{version}.json").write_text(json.dumps(request))
    receipt = {"submission_id": submission_id, "request_sha256": lg.canonical_sha(request), **RECORDED,
               **receipt_fields}
    (directory / f"receipt-{version}.json").write_text(json.dumps(receipt))
    return directory / f"receipt-{version}.json"


def add(ledger, tmp_path, receipt, run):
    path = tmp_path / f"{run['run_id']}.json"
    path.write_text(json.dumps(run))
    return lg.ledger_add(ledger, receipt, path)


def two_entries(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", {"industrial-physical-systems": 2,
                                                                              "natural-science": 1}))
    add(ledger, tmp_path, write_receipt(tmp_path, "v2", "profa-qe-arena-v2"),
        run_record("run-2", "profa-qe-arena-v2", {"office-white-collar": 2}))
    return ledger


# ---------------------------------------------------------------- records

def test_parse_result_reads_real_leaderboard_records():
    for run in board()["runs"]:
        run_id, domains = lg.parse_result(run)
        assert run_id == run["run_id"] and list(domains) == list(lg.DOMAIN_IDS)
        assert sum(d["solved"] for d in domains.values()) == run["solved"]


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.update(solved=r["solved"] + 1), "disagrees"),
    (lambda r: r.pop("submission_id"), "missing"),
    (lambda r: r.pop("benchmark"), "missing"),
    (lambda r: r["scores"].update(sports={"solved": 0, "total": 5}), "exactly the 8"),
    (lambda r: r["scores"].pop("cybersecurity"), "exactly the 8"),  # review r3 C: partial-domain record
    (lambda r: r["scores"]["cybersecurity"].update(solved=6), "solved <= total"),
])
def test_parse_result_rejects_malformed_records(mutate, message):
    run = board()["runs"][0]
    mutate(run)
    with pytest.raises(SystemExit, match=message):
        lg.parse_result(run)


# ---------------------------------------------------------------- integrity

def test_ledger_is_append_only_and_anchored(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    first = add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", {"natural-science": 3}))
    assert first["source"] == "manual" and first["request_sha256"].startswith("e6a38411")
    before = ledger.read_bytes()
    add(ledger, tmp_path, write_receipt(tmp_path, "v2", "profa-qe-arena-v2"),
        run_record("run-2", "profa-qe-arena-v2", {"office-white-collar": 2}))
    after = ledger.read_bytes()
    assert after.startswith(before) and after.count(b"\n") == 2
    with pytest.raises(SystemExit, match="never replaced"):
        add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", {}))
    assert ledger.read_bytes() == after
    assert [e["seq"] for e in lg.read_ledger(ledger)] == [1, 2]
    assert json.loads(lg.head_path(ledger).read_text())["seq"] == 2


@pytest.mark.parametrize("damage, message", [
    (lambda lines: lines[:-1], "head anchor"),  # review r3 D: drop the last line
    (lambda lines: lines + [""], "blank or corrupt"),  # review r3 E: trailing blank line
    (lambda lines: [lines[0].replace('"solved":3', '"solved":4', 1)] + lines[1:], "chain broken|disagrees"),
    (lambda lines: lines[:1] + ["{not json"] + lines[1:], "blank or corrupt"),
])
def test_damaged_ledgers_fail_cleanly(tmp_path, damage, message):
    ledger = two_entries(tmp_path)
    lines = ledger.read_text().splitlines()
    ledger.write_text("\n".join(damage(lines)) + "\n")
    with pytest.raises(SystemExit, match=message):
        lg.read_ledger(ledger)


def test_editing_the_last_entry_fails(tmp_path):
    # review r9: rewrite the last entry's score with a consistent record and no chain to break
    ledger = two_entries(tmp_path)
    lines = ledger.read_text().splitlines()
    last = json.loads(lines[-1])
    last["record"]["scores"]["office-white-collar"]["solved"] = 5
    last["record"]["solved"] = last["solved"] = 5
    last["domains"]["office-white-collar"]["solved"] = 5
    lines[-1] = json.dumps(last, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")
    with pytest.raises(SystemExit, match="head anchor"):
        lg.read_ledger(ledger)


def test_an_entry_must_agree_with_its_own_record(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", {"natural-science": 1}))
    entry = json.loads(ledger.read_text())
    entry["solved"] = 9  # a re-hashed edit that leaves the record alone
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    ledger.write_text(line + "\n")
    lg.head_path(ledger).write_text(json.dumps({"seq": 1, "sha256": lg.line_sha(line)}))
    with pytest.raises(SystemExit, match="disagrees with its own run record"):
        lg.read_ledger(ledger)


def test_sequence_numbers_are_checked(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", {}))
    entry = json.loads(ledger.read_text())
    entry["seq"] = 7  # a valid chain link with the wrong position
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    ledger.write_text(line + "\n")
    lg.head_path(ledger).write_text(json.dumps({"seq": 7, "sha256": lg.line_sha(line)}))
    with pytest.raises(SystemExit, match="chain broken"):
        lg.read_ledger(ledger)


@pytest.mark.parametrize("fields, message", [
    ({"state": "failed", "post_attempted": False}, "only a recorded submission"),  # review r3 A
    ({"post_attempted": False}, "only a recorded submission"),
])
def test_receipt_must_be_recorded(tmp_path, fields, message):
    receipt = write_receipt(tmp_path, "v1", "profa-qe-arena-v1", **fields)
    with pytest.raises(SystemExit, match=message):
        add(tmp_path / "l.jsonl", tmp_path, receipt, run_record("x", "profa-qe-arena-v1", {}))


def test_request_and_receipt_must_name_the_same_submission(tmp_path):
    receipt = write_receipt(tmp_path, "v1", "profa-qe-arena-v1")
    request = json.loads((tmp_path / "request-v1.json").read_text())
    request["submission_id"] = "someone-else"
    (tmp_path / "request-v1.json").write_text(json.dumps(request))
    data = json.loads(receipt.read_text())
    data["request_sha256"] = lg.canonical_sha(request)
    receipt.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="different submission ids"):
        add(tmp_path / "l.jsonl", tmp_path, receipt, run_record("x", "profa-qe-arena-v1", {}))


def test_ledger_refuses_a_run_for_another_submission(tmp_path):
    with pytest.raises(SystemExit, match="run is for"):
        add(tmp_path / "l.jsonl", tmp_path, RECEIPT, board()["runs"][0])


def test_ledger_refuses_a_request_that_does_not_match_the_receipt(tmp_path):
    request = json.loads((ROOT / "submission" / "request-v1.json").read_text())
    request["tasks"] = request["tasks"][:1]
    (tmp_path / "request-v1.json").write_text(json.dumps(request))
    (tmp_path / "receipt-v1.json").write_text(RECEIPT.read_text())
    with pytest.raises(SystemExit, match="hashes to"):
        add(tmp_path / "l.jsonl", tmp_path, tmp_path / "receipt-v1.json", run_record("x", "profa-qe-arena-v1", {}))


# ---------------------------------------------------------------- fetch

def ruv_receipts(directory):
    for i, run in enumerate(r for r in board()["runs"] if r["user"] == "ruv"):
        write_receipt(directory, f"v{i}", run["submission_id"])


def test_fetch_appends_matching_runs_once_and_versions_the_baseline(tmp_path):
    ruv_receipts(tmp_path)
    ledger, calls = tmp_path / "ledger.jsonl", []
    get = lambda url: calls.append(url) or board()  # noqa: E731 - the network, replaced by the saved response
    first = lg.ledger_fetch(ledger, "ruv", receipts=tmp_path, get=get)
    assert calls == [lg.LEADERBOARD] and first["matched"] == 2 and len(first["appended"]) == 2
    assert all(e["source"] == "fetched" and e["record"]["user"] == "ruv" for e in lg.read_ledger(ledger))
    before = ledger.read_bytes()
    again = lg.ledger_fetch(ledger, "ruv", receipts=tmp_path, board=board())
    assert again["appended"] == [] and len(again["already"]) == 2 and ledger.read_bytes() == before
    baseline, source = lg.resolve_baseline(ledger, None)
    assert baseline == lg.validate_scores(board()["baseline"]["scores"], "x") and "leaderboard" in source
    changed = board()
    changed["baseline"]["scores"]["natural-science"]["solved"] = 1  # review r8 T: keep both versions
    lg.ledger_fetch(ledger, "nobody", receipts=tmp_path, board=changed)
    versions = lg.baselines_path(ledger).read_text().splitlines()
    assert len(versions) == 2 and lg.resolve_baseline(ledger, None)[0]["natural-science"]["solved"] == 1


def test_fetch_filters_by_submission_and_skips_runs_without_a_receipt(tmp_path):
    target = next(r for r in board()["runs"] if r["user"] == "ruv")
    ledger = tmp_path / "new-dir" / "l.jsonl"
    report = lg.ledger_fetch(ledger, "ruv", target["submission_id"], tmp_path, board=board())
    assert report["matched"] == 1 and report["appended"] == []
    assert report["no_receipt"] == [f"{target['submission_id']}/{target['run_id']}"]
    assert not ledger.exists() and lg.baselines_path(ledger).exists()


def test_fetch_user_match_is_case_sensitive_with_a_warning(tmp_path, capsys):
    report = lg.ledger_fetch(tmp_path / "l.jsonl", "RUV", receipts=tmp_path, board=board())  # review r8 U
    assert report["matched"] == 0 and report["case_mismatch"]
    assert "case-sensitively" in capsys.readouterr().err


def test_fetch_refuses_unverified_pagination():
    paged = dict(board(), next_cursor="12-34")
    with pytest.raises(SystemExit, match="unverified"):
        lg.fetch_leaderboard(lambda url: paged)


def test_fetch_with_a_cursor_template_follows_and_guards():
    template = "{url}?cursor={cursor}"
    pages = {lg.LEADERBOARD: dict(board(), next_cursor="1-2"),
             lg.LEADERBOARD + "?cursor=1-2": {"runs": [{"run_id": "extra"}], "next_cursor": None}}
    merged = lg.fetch_leaderboard(pages.__getitem__, cursor_url=template)
    assert merged["runs"][-1] == {"run_id": "extra"} and len(merged["runs"]) == 4
    calls = []
    with pytest.raises(SystemExit, match="refusing leaderboard cursor"):
        lg.fetch_leaderboard(lambda url: calls.append(url) or dict(board(), next_cursor="../admin"),
                             cursor_url=template)
    assert calls == [lg.LEADERBOARD]  # the bad cursor is never requested
    loop = {lg.LEADERBOARD: dict(board(), next_cursor="1-2"),  # review r8 Q
            lg.LEADERBOARD + "?cursor=1-2": {"runs": [], "next_cursor": "1-2"}}
    with pytest.raises(SystemExit, match="repeated cursor"):
        lg.fetch_leaderboard(loop.__getitem__, cursor_url=template)
    counter = iter(range(1, 1000))
    endless = lambda url: dict(board(), next_cursor=f"1-{next(counter)}")  # noqa: E731 - distinct cursors forever
    with pytest.raises(SystemExit, match="still paging after 3"):
        lg.fetch_leaderboard(endless, cursor_url=template, max_pages=3)
    bad_page = {lg.LEADERBOARD: dict(board(), next_cursor="1-2"),  # review r8 P
                lg.LEADERBOARD + "?cursor=1-2": {"runs": {"a": 1}, "next_cursor": None}}
    with pytest.raises(SystemExit, match="no runs list"):
        lg.fetch_leaderboard(bad_page.__getitem__, cursor_url=template)


def test_fetch_caps_the_response_size(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            return b"[" * n

    monkeypatch.setattr(lg.urllib.request, "urlopen", lambda request, timeout: Response())
    with pytest.raises(SystemExit, match="over 1024 bytes"):
        lg.http_get_json(lg.LEADERBOARD, max_bytes=1024)


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["office-white-collar"].update(total=10), "office-white-collar.*the baseline 10"),  # review r8 R
    (lambda s: s["office-white-collar"].update(solved="2"), "integers"),  # review r8 S
    (lambda s: s.pop("cybersecurity"), "exactly the 8"),
])
def test_baseline_is_validated(mutate, message):
    b = board()
    mutate(b["baseline"]["scores"])
    entries = [{"run_id": r["run_id"], "domains": lg.parse_result(r)[1]} for r in board()["runs"]]
    with pytest.raises(SystemExit, match=message):
        lg.null_domains(entries, lg.validate_scores(b["baseline"]["scores"], "baseline"))


def test_offline_fallback_baseline_matches_the_leaderboard(tmp_path):
    baseline, source = lg.resolve_baseline(tmp_path / "ledger.jsonl", None)
    assert "OFFLINE FALLBACK" in source
    assert baseline == lg.validate_scores(board()["baseline"]["scores"], "x") == lg.parse_baseline(str(BOARD))


# ---------------------------------------------------------------- the null

DOMAINS = {"industrial": (5, 2), "office": (5, 2), **{f"zero{j}": (5, 0) for j in range(6)}}


def test_plugin_null_hand_worked():
    # Two domains of 2 tasks, baseline 1/2, k = 2. Bin(2, .5): F(0)=1/4, F(1)=3/4, so
    # E[max] = (1 - (1/4)^2) + (1 - (3/4)^2) = 11/8 per domain; 11/4 for the composite.
    dist = null.composite_pmf(2, {"a": (2, 1), "b": (2, 1)}, "plugin")
    assert null.mean(dist) == pytest.approx(2.75)
    assert null.tail(dist, 4) == pytest.approx((7 / 16) ** 2)


def test_model_null_single_run_is_the_beta_binomial_mean():
    # k = 1: each domain is BetaBinomial(n, s + .5, n - s + .5) with mean n (s + .5) / (n + 1)
    assert null.mean(null.composite_pmf(1, DOMAINS)) == pytest.approx(2 * 5 * 2.5 / 6 + 6 * 5 * 0.5 / 6)
    assert null.mean(null.composite_pmf(1, DOMAINS, "plugin")) == pytest.approx(4.0)


def test_model_null_matches_an_independent_integration():
    # E[max of 3] for one 2/5 domain: integrate F(x; 5, p)^3 against Beta(2.5, 3.5) on a fine grid
    p = (np.arange(200_000) + 0.5) / 200_000
    density = p ** 1.5 * (1 - p) ** 2.5
    density /= density.sum()
    cdf = np.cumsum([[math.comb(5, x) * q ** x * (1 - q) ** (5 - x) for x in range(6)] for q in p[::50]], axis=1)
    expected = sum(1 - (density.reshape(-1, 50).sum(axis=1) @ cdf[:, x - 1] ** 3) for x in range(1, 6))
    assert null.mean(null.max_pmf(5, 3, solved=2)) == pytest.approx(expected, abs=2e-3)


def test_sensitivity_grows_with_the_zero_domains_rate():
    means = [null.mean(null.composite_pmf(3, DOMAINS, p_zero=z)) for z in (0.0, 0.02, 0.05)]
    assert means == sorted(means) and means[0] < null.mean(null.composite_pmf(3, DOMAINS))


def test_show_labels_the_composite_against_the_model_null(tmp_path, capsys):
    ledger = two_entries(tmp_path)
    entries = lg.read_ledger(ledger)
    baseline = lg.parse_baseline(lg.BASELINE)
    empty = report.ledger_show(entries, baseline)
    assert empty == {} and "2 manual entries left out" in capsys.readouterr().out
    summary = report.ledger_show(entries, baseline, source="test", include_manual=True,
                             head=json.loads(lg.head_path(ledger).read_text()))
    printed = capsys.readouterr().out
    domains = lg.null_domains(entries, baseline)
    model = null.composite_pmf(2, domains)
    assert summary["composite"] == 5 and summary["model_expected"] == pytest.approx(null.mean(model))
    assert summary["model_p_value"] == pytest.approx(null.tail(model, 5)) and summary["empirical"] is None
    single = null.composite_pmf(1, domains)
    assert {r: v["model_p"] for r, v in summary["per_run"].items()} == {
        "run-1": pytest.approx(null.tail(single, 3)), "run-2": pytest.approx(null.tail(single, 2))}
    assert f"{null.tail(single, 3):.3f}" in printed and "model-based null" in printed
    assert "RANK AMONG OTHER ENTRANTS' RUNS unavailable" in printed and "sensitivity, model-based" in printed
    assert "p_zero=0.05" in printed and "lower reference" in printed
    assert "ledger head: seq 2" in printed and "selection-inflated" in printed
    # review v2 N4: the comparison with the untrained baseline comes first
    assert printed.index("AGAINST THE UNTRAINED BASELINE") < printed.index("PER-RUN SCORES")
    assert summary["per_run"]["run-1"]["vs_baseline"] == 3 - 4
    assert summary["per_run"]["run-1"]["vs_baseline_domains"]["natural-science"] == 1
    assert summary["per_run"]["run-2"]["vs_baseline_domains"]["industrial-physical-systems"] == -2


def test_show_refuses_to_mix_benchmarks(tmp_path):
    # review r3 B
    ledger = tmp_path / "ledger.jsonl"
    add(ledger, tmp_path, RECEIPT, run_record("run-1", "profa-qe-arena-v1", {}))
    add(ledger, tmp_path, write_receipt(tmp_path, "v2", "profa-qe-arena-v2"),
        run_record("run-2", "profa-qe-arena-v2", {"office-white-collar": 4}, benchmark="heldout-v3"))
    with pytest.raises(SystemExit, match="span benchmarks"):
        report.ledger_show(lg.read_ledger(ledger), lg.parse_baseline(lg.BASELINE), include_manual=True)


# ---------------------------------------------------------------- the empirical null

def tiny_population():
    """Two other runs: A solves 1 industrial task, B 1 office task, nothing else."""
    def other(run_id, scores):
        return dict(run_record(run_id, f"sub-{run_id}", scores, user=f"user-{run_id}"))
    runs = [other("A", {"industrial-physical-systems": 1}), other("B", {"office-white-collar": 1})]
    return empirical.population(runs, "heldout-v2", {"Profa-QE"}, lg.parse_result)


def test_empirical_null_hand_worked():
    pop = tiny_population()
    s = empirical.summary(pop)
    assert (s["runs"], s["mean"], s["min"], s["max"]) == (2, 1.0, 1, 1)
    assert s["rates"]["industrial-physical-systems"] == pytest.approx(1 / 10)  # 1 solve over 2 runs x 5 tasks
    # one run: the composite is 1 either way; two DISTINCT runs are always A and B, so 2 (review v2 m-b:
    # with replacement, AA and BB gave 1 and the expectation was 1.5)
    one, two = empirical.composite_null(pop, 1, 1), empirical.composite_null(pop, 2, 2)
    assert (one["expected"], one["p_ge"]) == (1.0, 1.0) and "exact" in one["method"]
    assert (two["expected"], two["p_ge"]) == (2.0, 1.0)
    with pytest.raises(SystemExit, match="3 distinct runs from 2"):
        empirical.composite_null(pop, 3, 2)
    assert empirical.run_position(pop, 1) == {"percentile": 50.0, "p_other_ge": 1.0}
    assert empirical.run_position(pop, 2) == {"percentile": 100.0, "p_other_ge": 0.0}
    deltas = empirical.domain_deltas(pop, lg.parse_result(run_record("x", "s", {"industrial-physical-systems": 2}))[1])
    assert deltas["industrial-physical-systems"] == pytest.approx(1.5) and deltas["office-white-collar"] == -0.5


def test_empirical_monte_carlo_agrees_with_the_closed_form():
    # 40 other runs: 6 solve one industrial task, 10 one office task, 24 nothing. k = 6 distinct runs is past
    # the exact limit (C(40,6) > 200k). E = 2 - P(no industrial run) - P(no office run), hypergeometric.
    runs = [run_record(f"r{i}", f"s{i}", {"industrial-physical-systems": 1} if i < 6 else
                       {"office-white-collar": 1} if i < 16 else {}, user=f"u{i}") for i in range(40)]
    pop = empirical.population(runs, "heldout-v2", set(), lg.parse_result)
    result = empirical.composite_null(pop, 6, 2)
    assert "Monte Carlo" in result["method"]
    none = lambda bad: math.comb(40 - bad, 6) / math.comb(40, 6)  # noqa: E731
    assert result["expected"] == pytest.approx(2 - none(6) - none(10), abs=0.01)
    assert result["p_ge"] == pytest.approx((1 - none(6)) * 1 - (none(10) - none(16)), abs=0.01)


def test_population_filters_benchmark_users_runs_and_non_objects():
    # review v2 m-g: a heldout-v3 run must not enter a heldout-v2 population; m-e: non-dict entries
    runs = [run_record("a", "sa", {"office-white-collar": 1}, user="u1"),
            run_record("b", "sb", {"office-white-collar": 5}, user="u2", benchmark="heldout-v3"),
            run_record("c", "sc", {"office-white-collar": 5}, user="me"),
            run_record("d", "sd", {"office-white-collar": 5}, user="u3"), "junk"]
    pop = empirical.population(runs, "heldout-v2", {"me"}, lg.parse_result, exclude_runs={"d"})
    assert pop["runs"] == ["a"] and pop["skipped"] == 1
    junk = dict(board(), runs=board()["runs"] + ["junk"])
    with pytest.raises(SystemExit, match="not an object"):
        lg.ledger_fetch(Path("/nonexistent/l.jsonl"), "Profa-QE", board=junk)


def test_empirical_bayes_prior_for_the_zero_domains():
    pop = tiny_population()  # pooled: industrial 1 of 10 tasks, office 1 of 10, the rest 0 of 10
    prior = empirical.zero_domain_prior(pop)
    assert prior["industrial-physical-systems"] == (1.5, 9.5) and prior["cybersecurity"] == (0.5, 10.5)
    domains = {"industrial-physical-systems": (5, 2), "cybersecurity": (5, 0)}
    one = null.composite_pmf(1, domains, zero_prior=prior)
    # k = 1: cyber is BetaBinomial(5, .5, 10.5 + 5), mean 5 * .5 / 16; industrial keeps Jeffreys from 2/5
    assert null.mean(one) == pytest.approx(5 * 0.5 / 16 + 5 * 2.5 / 6)


def test_empirical_null_on_the_saved_leaderboard():
    board = json.loads(FULL.read_text())
    pop = empirical.population(board["runs"], "heldout-v2", {"Profa-QE"}, lg.parse_result)
    s = empirical.summary(pop)
    others = [r for r in board["runs"] if r["user"] != "Profa-QE"]
    totals = [r["solved"] for r in others]
    assert s["runs"] == len(others) == 38 and s["mean"] == pytest.approx(sum(totals) / 38)
    assert s["rates"]["office-white-collar"] == pytest.approx(
        sum(r["scores"]["office-white-collar"]["solved"] for r in others) / (5 * 38))
    assert round(s["mean"], 2) == 2.95 and s["rates"]["mathematics-or-formal-reasoning"] == 0.0
    position = empirical.run_position(pop, 4)
    assert position["p_other_ge"] == pytest.approx(sum(t >= 4 for t in totals) / 38)
    assert position["percentile"] == pytest.approx(100 * (sum(t < 4 for t in totals) + sum(t == 4 for t in totals) / 2) / 38)


def test_show_reports_v1_against_other_entrants(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    (tmp_path / "receipt-v1.json").write_text(RECEIPT.read_text())
    (tmp_path / "request-v1.json").write_text((ROOT / "submission" / "request-v1.json").read_text())
    fetched = lg.ledger_fetch(ledger, "Profa-QE", receipts=tmp_path, board=json.loads(FULL.read_text()))
    assert fetched["appended"] == ["c583c91a8555ec252893d574"] and fetched["population"] == 39
    entries = lg.read_ledger(ledger)
    summary = report.ledger_show(entries, lg.resolve_baseline(ledger, None)[0], lg.load_population(ledger))
    printed = capsys.readouterr().out
    row = summary["per_run"]["c583c91a8555ec252893d574"]
    pop = empirical.population(json.loads(FULL.read_text())["runs"], "heldout-v2", {"Profa-QE"}, lg.parse_result)
    assert row["p_other_ge"] == empirical.run_position(pop, 4)["p_other_ge"]
    assert row["deltas"]["finance-economics"] == pytest.approx(1 - 5 * empirical.summary(pop)["rates"]["finance-economics"])
    assert summary["empirical"]["expected"] == pytest.approx(empirical.summary(pop)["mean"])  # k = 1
    assert row["vs_baseline"] == 0 and row["vs_baseline_domains"]["industrial-physical-systems"] == -1
    assert "RANK AMONG OTHER ENTRANTS' RUNS (not a test of whether training helped): 38 runs" in printed
    assert "excluding ['Profa-QE']" in printed and "upper bound on evaluation noise" in printed
    assert printed.index("AGAINST THE UNTRAINED BASELINE") < printed.index("RANK AMONG OTHER")
    assert printed.index("rank among other entrants: best-per-domain") < printed.index("sensitivity, model-based")
    assert summary["eb_expected"] < summary["model_expected"]


def test_cli_fetch_and_show_from_a_saved_board(tmp_path, capsys):
    (tmp_path / "receipt-v1.json").write_text(RECEIPT.read_text())
    (tmp_path / "request-v1.json").write_text((ROOT / "submission" / "request-v1.json").read_text())
    ledger = str(tmp_path / "ledger.jsonl")
    assert cli.main(["ledger", "fetch", "--user", "Profa-QE", "--ledger", ledger, "--receipts", str(tmp_path),
                     "--from-file", str(FULL)]) == 0
    assert cli.main(["ledger", "show", "--ledger", ledger]) == 0
    printed = capsys.readouterr().out
    assert "RANK AMONG OTHER ENTRANTS' RUNS" in printed and "c583c91a8555ec252893d574" in printed


def test_a_manual_entry_does_not_remove_its_user_from_the_population(tmp_path):
    # review v2 a_ledger: a hand-entered run in someone else's name used to drop that user's real runs
    ledger = tmp_path / "ledger.jsonl"
    (tmp_path / "receipt-v1.json").write_text(RECEIPT.read_text())
    (tmp_path / "request-v1.json").write_text((ROOT / "submission" / "request-v1.json").read_text())
    full = json.loads(FULL.read_text())
    lg.ledger_fetch(ledger, "Profa-QE", receipts=tmp_path, board=full)
    other = next(r for r in full["runs"] if r["user"] == "ruv")
    add(ledger, tmp_path, write_receipt(tmp_path, "v9", "profa-qe-arena-v9"),
        run_record("fake", "profa-qe-arena-v9", {"office-white-collar": 1}, user=other["user"]))
    summary = report.ledger_show(lg.read_ledger(ledger), lg.resolve_baseline(ledger, None)[0],
                                 lg.load_population(ledger), include_manual=True, out=open(os.devnull, "w"))
    assert summary["empirical"]["population"] == 38  # only Profa-QE is excluded, not ruv


def test_atomic_head_and_repair_after_an_interrupted_write(tmp_path):
    # review v2 a_ledger: a crash between the append and the head write
    ledger = two_entries(tmp_path)
    lg.head_path(ledger).unlink()
    with pytest.raises(SystemExit, match="repair-head"):
        lg.read_ledger(ledger)
    with pytest.raises(SystemExit, match="--i-checked-the-ledger"):
        lg.repair_head(ledger, confirmed=False)
    assert lg.repair_head(ledger, confirmed=True)["seq"] == 2 and len(lg.read_ledger(ledger)) == 2
    assert not lg.head_path(ledger).with_name(lg.head_path(ledger).name + ".tmp").exists()
    assert cli.main(["ledger", "repair-head", "--ledger", str(ledger), "--i-checked-the-ledger"]) == 0
    ledger.write_text(ledger.read_text().replace('"solved":3', '"solved":4', 1))  # repair never hides damage
    with pytest.raises(SystemExit):
        lg.repair_head(ledger, confirmed=True)


def test_a_deliberate_consistent_rewrite_is_not_detected(tmp_path):
    # documented limit (review v2 m-c): the chain and head catch accidents, not someone with write access
    ledger = two_entries(tmp_path)
    lines = ledger.read_text().splitlines()
    last = json.loads(lines[-1])
    last["record"]["scores"]["office-white-collar"]["solved"] = 5
    last["record"]["solved"] = last["solved"] = 5
    last["domains"]["office-white-collar"]["solved"] = 5
    lines[-1] = json.dumps(last, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n")
    lg.write_head(ledger, 2, lg.line_sha(lines[-1]))
    assert lg.read_ledger(ledger)[-1]["solved"] == 5
    assert "catch ACCIDENTS" in lg.__doc__
