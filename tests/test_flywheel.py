"""Flywheel tooling: posterior math, grouping, selection, the ledger's null, and one-knob decisions."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flywheel as fw  # noqa: E402

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
    assert fw.beta_cdf(x, 1, 1) == pytest.approx(x, abs=1e-12)
    assert fw.beta_cdf(x, 3.5, 1) == pytest.approx(x ** 3.5, abs=1e-12)
    assert fw.beta_cdf(x, 1, 2.5) == pytest.approx(1 - (1 - x) ** 2.5, abs=1e-12)
    assert fw.beta_cdf(x, 2, 2) == pytest.approx(3 * x ** 2 - 2 * x ** 3, abs=1e-12)
    # Jeffreys prior Beta(0.5, 0.5) is the arcsine distribution
    assert fw.beta_cdf(x, 0.5, 0.5) == pytest.approx(2 / math.pi * math.asin(math.sqrt(x)), abs=1e-12)


@pytest.mark.parametrize("q", [0.05, 0.25, 0.5, 0.95])
def test_beta_ppf_inverts_closed_forms(q):
    assert fw.beta_ppf(q, 0.5, 0.5) == pytest.approx(math.sin(math.pi * q / 2) ** 2, abs=1e-12)
    assert fw.beta_ppf(q, 4, 1) == pytest.approx(q ** 0.25, abs=1e-12)
    assert fw.beta_ppf(q, 1, 3) == pytest.approx(1 - (1 - q) ** (1 / 3), abs=1e-12)


def test_beta_cdf_is_symmetric():
    for a, b, x in [(15.5, 1.5, 0.8), (0.5, 73.5, 0.03), (142.5, 4.5, 0.97)]:
        assert fw.beta_cdf(x, a, b) == pytest.approx(1 - fw.beta_cdf(1 - x, b, a), abs=1e-12)


def test_jeffreys_posterior():
    p = fw.posterior(15, 16)
    assert (p["a"], p["b"]) == (15.5, 1.5)
    assert p["mean"] == pytest.approx(15.5 / 17)
    assert fw.beta_cdf(p["lo"], 15.5, 1.5) == pytest.approx(0.05, abs=1e-9)
    assert fw.beta_cdf(p["hi"], 15.5, 1.5) == pytest.approx(0.95, abs=1e-9)
    empty = fw.posterior(0, 0)  # the prior itself: arcsine quantiles
    assert empty["lo"] == pytest.approx(math.sin(math.pi * 0.05 / 2) ** 2, abs=1e-12)
    assert empty["hi"] == pytest.approx(math.sin(math.pi * 0.95 / 2) ** 2, abs=1e-12)


def test_projected_informative_closed_forms():
    # p ~ Uniform: E[p^2] = E[(1-p)^2] = 1/3
    assert fw.projected_informative(1, 1, 2) == pytest.approx(1 / 3)
    # a near point mass at p=0.3
    assert fw.projected_informative(3e6, 7e6, 5) == pytest.approx(1 - 0.3 ** 5 - 0.7 ** 5, abs=1e-6)


# ---------------------------------------------------------------- grouping and error rows

def test_groups_are_task_seed_corpus():
    rows = [row("qe_bugfix-d2", 1, 0, 1.0, "f"), row("qe_bugfix-d2", 1, 1, 1.0, "f"),
            row("qe_bugfix-d2", 1, 0, 0.0, "g"), row("qe_bugfix-d2", 1, 1, 0.0, "g")]
    s = fw.summarize(rows)
    assert s["groups"] == 2 and s["informative"]["k"] == 0  # pooled over corpus it would be 1/1
    pooled = fw.summarize(rows, by_corpus=False)
    assert pooled["groups"] == 1 and pooled["informative"]["k"] == 1


def test_groups_split_by_seed_and_task():
    rows = [row("qe_bugfix-d2", 1, 0, 1.0), row("qe_bugfix-d2", 2, 0, 0.0),
            row("qe_bugfix-d2", 1, 1, 1.0), row("qe_bugfix-d2", 2, 1, 0.0),
            row("qe_bugfix-d3", 1, 0, 0.0), row("qe_bugfix-d3", 1, 1, 0.5)]
    stats = fw.compute_stats(rows)["cells"]
    assert stats["qe_bugfix-d2"]["groups"] == 2 and stats["qe_bugfix-d2"]["informative"]["k"] == 0
    assert stats["qe_bugfix-d3"]["groups"] == 1 and stats["qe_bugfix-d3"]["informative"]["k"] == 1


def test_equal_partial_rewards_are_not_informative():
    s = fw.summarize([row("qe_mutant_kill-d3", 1, 0, 2 / 3), row("qe_mutant_kill-d3", 1, 1, 0.6666666666666666)])
    assert s["informative"]["k"] == 0 and s["solve"]["k"] == 0


def test_error_rows_are_excluded_not_zeroed():
    rows = [row("qe_bugfix-d2", 1, 0, 1.0), row("qe_bugfix-d2", 1, 1, None, error="HTTP 500"),
            row("qe_bugfix-d2", 2, 0, 1.0), row("qe_bugfix-d2", 2, 1, 1.0)]
    s = fw.summarize(rows)
    assert (s["episodes"], s["errors"], s["valid"]) == (4, 1, 3)
    assert s["mean_reward"] == 1.0  # zeroing would give 0.75
    assert (s["solve"]["k"], s["solve"]["n"]) == (3, 3)
    assert (s["groups"], s["short_groups"]) == (1, 1)  # seed 1 has one valid episode left
    assert s["informative"]["k"] == 0  # zeroing would make seed 1 look informative
    assert s["ends"] == {"error": 1, "submit": 3}


def test_real_fixture_counts():
    rows = fw.load_runs(sorted(FIXTURES.glob("*.jsonl")))
    assert all("messages" not in r for r in rows)
    stats = fw.compute_stats(rows)
    kill = stats["cells"]["qe_mutant_kill-d1"]  # seeds 830000-3: [1,.5] [0,0] [1,1] [1,1]
    assert (kill["solve"]["k"], kill["solve"]["n"], kill["informative"]["k"], kill["groups"]) == (5, 8, 1, 4)
    dead = stats["cell_corpus"]["qe_bugfix-d2|alternate_chat_roles"]
    assert (dead["errors"], dead["valid"], dead["groups"]) == (4, 0, 0)
    split = stats["cell_corpus"]["qe_bugfix-d2|flaky_run_summary"]  # 2/4 solved, but per seed 2/2 and 0/2
    assert split["solve"]["k"] == 2 and split["informative"]["k"] == 0


# ---------------------------------------------------------------- select

def synthetic_stats(cells: dict[str, list[tuple[int, list[float]]]]) -> dict:
    rows = [row(t, seed, a, r) for t, groups in cells.items() for seed, rewards in groups
            for a, r in enumerate(rewards)]
    return fw.compute_stats(rows)


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
    tasks, reasons = fw.select_cells(synthetic_stats(MIXED), template(), max_cells=6, seed=1)
    by_id = {r["task_id"]: r for r in reasons}
    assert {t["task_id"] for t in tasks} == {"qe_mutant_kill-d1", "qe_bugfix-d2"}
    assert by_id["qe_mutant_kill-d2"]["reason"].startswith("saturated")
    assert by_id["qe_mutant_kill-d3"]["reason"].startswith("dead")
    assert by_id["qe_bugfix-d3"]["reason"].startswith("saturated")  # 15/16 -> 0.912
    loose, _ = fw.select_cells(synthetic_stats(MIXED), template(), 6, 1, saturated=0.95)
    assert "qe_bugfix-d3" in {t["task_id"] for t in loose}


def test_select_is_deterministic_and_respects_max_cells():
    stats = synthetic_stats({e["task_id"]: [(s, [1.0, 0.0]) for s in range(3)] for e in template()})
    first = fw.select_cells(stats, template(), max_cells=2, seed=42)
    assert first == fw.select_cells(stats, template(), max_cells=2, seed=42)
    assert len(first[0]) == 2
    assert sum(r["decision"] == "selected" for r in first[1]) == 2


def test_select_preserves_every_template_field():
    tmpl = template()
    original = json.loads(json.dumps(tmpl))
    tasks, _ = fw.select_cells(synthetic_stats(MIXED), tmpl, max_cells=6, seed=3)
    by_id = {e["task_id"]: e for e in original}
    for task in tasks:
        assert task == by_id[task["task_id"]]
    tasks[0]["completion_tokens"] = 1
    assert tmpl == original  # output is a copy


def test_select_records_cells_without_a_template_entry():
    stats = synthetic_stats({"qe_bugfix-d1": [(0, [1.0, 0.0])]})
    tasks, reasons = fw.select_cells(stats, template(), 6, 1)
    by_id = {r["task_id"]: r for r in reasons}
    assert not tasks
    assert "no template entry" in by_id["qe_bugfix-d1"]["reason"]
    assert by_id["qe_mutant_kill-d1"]["reason"] == "no calibration data"


def test_select_cli_writes_identical_files_and_sidecar(tmp_path):
    runs = sorted(str(p) for p in FIXTURES.glob("*.jsonl"))
    outs = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.json"
        assert fw.main(["select", *runs, "--max-cells", "6", "--seed", "7", "--template", str(TEMPLATE),
                        "--out", str(out)]) == 0
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()
    why = json.loads((tmp_path / "a.why.json").read_text())
    assert why["seed"] == 7 and len(why["runs"]) == 2 and all(c.get("reason") for c in why["cells"])


# ---------------------------------------------------------------- ledger

def result_file(tmp_path, name, run_id, industrial, office, total=2):
    path = tmp_path / name
    path.write_text(json.dumps({"run_id": run_id, "domains": {
        "industrial": {"solved": industrial, "total": total}, "office": {"solved": office, "total": total}}}))
    return path


def second_receipt(tmp_path):
    request = json.loads((ROOT / "submission" / "request-v1.json").read_text())
    request["submission_id"] = "profa-qe-arena-v2"
    (tmp_path / "request-v2.json").write_text(json.dumps(request))
    receipt = {"submission_id": "profa-qe-arena-v2", "request_sha256": fw.canonical_sha(request)}
    (tmp_path / "receipt-v2.json").write_text(json.dumps(receipt))
    return tmp_path / "receipt-v2.json"


def test_ledger_is_append_only(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    first = fw.ledger_add(ledger, RECEIPT, result_file(tmp_path, "r1.json", "run-1", 2, 1), now="t1")
    assert first["cells"][0] == "qe_mutant_kill-d1" and first["request_sha256"].startswith("e6a38411")
    before = ledger.read_bytes()
    fw.ledger_add(ledger, second_receipt(tmp_path), result_file(tmp_path, "r2.json", "run-2", 0, 2), now="t2")
    after = ledger.read_bytes()
    assert after.startswith(before) and after.count(b"\n") == 2
    with pytest.raises(SystemExit, match="never replaced"):
        fw.ledger_add(ledger, RECEIPT, result_file(tmp_path, "r3.json", "run-3", 2, 2))
    assert ledger.read_bytes() == after
    entries = fw.read_ledger(ledger)
    assert [e["seq"] for e in entries] == [1, 2]
    ledger.write_bytes(after.replace(b'"solved":3', b'"solved":4', 1))
    with pytest.raises(SystemExit, match="chain broken"):
        fw.read_ledger(ledger)


def test_ledger_refuses_a_request_that_does_not_match_the_receipt(tmp_path):
    request = json.loads((ROOT / "submission" / "request-v1.json").read_text())
    request["tasks"] = request["tasks"][:1]
    (tmp_path / "request-v1.json").write_text(json.dumps(request))
    (tmp_path / "receipt-v1.json").write_text(RECEIPT.read_text())
    with pytest.raises(SystemExit, match="hashes to"):
        fw.ledger_add(tmp_path / "l.jsonl", tmp_path / "receipt-v1.json", result_file(tmp_path, "r.json", "x", 1, 1))


def test_composite_null_hand_worked():
    # Two domains of 2 tasks, baseline p = 1/2, k = 2 runs. Bin(2, .5) has F(0)=1/4, F(1)=3/4, so
    # E[max] = (1 - (1/4)^2) + (1 - (3/4)^2) = 15/16 + 7/16 = 11/8 per domain; 11/4 for the composite.
    dist = fw.null_composite(2, {"industrial": (2, 0.5), "office": (2, 0.5)})
    assert float(np.dot(np.arange(len(dist)), dist)) == pytest.approx(2.75)
    assert fw.tail(dist, 4) == pytest.approx((7 / 16) ** 2)  # both maxes hit 2: P(max=2) = 1 - (3/4)^2
    one = fw.null_composite(1, {"industrial": (20, 0.1), "office": (20, 0.1)})  # k=1 is just the baseline
    assert float(np.dot(np.arange(len(one)), one)) == pytest.approx(4.0)


def test_ledger_show_labels_composite_against_null(tmp_path, capsys):
    ledger = tmp_path / "ledger.jsonl"
    fw.ledger_add(ledger, RECEIPT, result_file(tmp_path, "r1.json", "run-1", 2, 0))
    fw.ledger_add(ledger, second_receipt(tmp_path), result_file(tmp_path, "r2.json", "run-2", 0, 2))
    summary = fw.ledger_show(fw.read_ledger(ledger), {"industrial": 1, "office": 1})
    printed = capsys.readouterr().out
    assert summary["composite"] == 4 and summary["best"] == {"industrial": 2, "office": 2}
    assert summary["null_expected"] == pytest.approx(2.75)
    assert summary["null_p_value"] == pytest.approx(49 / 256)
    assert "PER-RUN SCORES" in printed and "selection-inflated" in printed


# ---------------------------------------------------------------- onecell-search

def config():
    return fw.init_config(template())


def test_init_config_reads_environment_and_template():
    c = config()
    assert c["max_steps"] == 12 and c["max_test_runs"] == 3
    assert c["difficulty_mix"] == {"d1": 0.2, "d2": 0.4, "d3": 0.4}
    assert set(c["completion_tokens"]) == {e["task_id"] for e in template()}


@pytest.mark.parametrize("seed", range(40))
def test_propose_changes_exactly_one_knob(seed):
    seen = set(range(830_000, 830_008))
    plan = fw.propose(config(), seed, seen, [], sorted(config()["completion_tokens"]))
    assert fw.changed_knobs(plan["baseline"], plan["candidate"]) == [plan["knob"]]
    assert plan["from"] != plan["to"]
    disc, conf = set(plan["discovery"]["seeds"]), set(plan["confirmation"]["seeds"])
    assert not disc & conf and not conf & seen and not disc & seen
    assert plan == fw.propose(config(), seed, seen, [], sorted(config()["completion_tokens"]))


def test_mix_change_counts_as_one_knob():
    c = config()
    moved = fw.apply_move(c, "difficulty_mix", {"d1": 0.1, "d2": 0.5, "d3": 0.4})
    assert fw.changed_knobs(c, moved) == ["difficulty_mix"]


def test_propose_skips_tried_moves():
    c = config()
    history = [{"knob": k, "to": v} for k, v in fw.candidate_moves(c)[1:]]
    plan = fw.propose(c, 0, set(), history, ["qe_bugfix-d2"])
    assert (plan["knob"], plan["to"]) == fw.candidate_moves(c)[0]


def split_rows(split, reward_of):
    return [row(t, s, a, reward_of(t, s, a)) for t in split["task_ids"] for s in split["seeds"]
            for a in range(split["attempts"])]


def make_plan():
    c = config()
    c["difficulty_mix"] = {"d1": 0.0, "d2": 1.0, "d3": 0.0}
    plan = fw.propose(c, 5, set(), [], ["qe_bugfix-d2"], discovery_seeds=4, confirm_seeds=4, attempts=2)
    if plan["knob"] == "difficulty_mix":  # keep the reweighting out of this test
        plan = fw.propose(c, 6, set(), [], ["qe_bugfix-d2"], discovery_seeds=4, confirm_seeds=4, attempts=2)
    assert plan["knob"] != "difficulty_mix"
    return plan


def decide_with(plan, base_d, cand_d, base_c, cand_c):
    return fw.decide(plan, split_rows(plan["discovery"], base_d), split_rows(plan["discovery"], cand_d),
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
        fw.decide(plan, base, base[:-1], split_rows(plan["confirmation"], half),
                  split_rows(plan["confirmation"], half))
    with pytest.raises(SystemExit, match="outside the plan"):
        fw.decide(plan, base, base, base, base)  # discovery rows used as confirmation


def test_decide_drops_error_pairs():
    plan = make_plan()
    half = lambda t, s, a: float(a == 0)  # noqa: E731
    seed0 = plan["discovery"]["seeds"][0]
    errs = lambda t, s, a: None if (s == seed0 and a == 1) else float(a == 0)  # noqa: E731
    result = decide_with(plan, half, errs, half, half)
    assert result["discovery"]["dropped_error_pairs"] == 1
    assert result["discovery"]["delta"] == pytest.approx(4 / 7 - 4 / 7)
