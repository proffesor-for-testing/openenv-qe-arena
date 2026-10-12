"""Append-only, hash-chained ledger of arena runs, and the exact null for best-per-domain composites.

A result is one ``runs[]`` record from the public leaderboard (GET /api/leaderboard, no auth)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GENESIS = "0" * 64
LEADERBOARD = "https://openenvarena-arena.hf.space/api/leaderboard"
DOMAIN_IDS = ("software-engineering", "industrial-physical-systems", "natural-science", "office-white-collar",
              "finance-economics", "mathematics-or-formal-reasoning", "media-content-production", "cybersecurity")
RUN_KEYS = ("run_id", "submission_id", "user", "scores", "solved", "total")
# Offline fallback only: the leaderboard's own `baseline` (Qwen3.8-27B 1d4bf0f, measured 2026-10-07) wins.
BASELINE = "industrial-physical-systems=2,office-white-collar=2"
KEPT = ("benchmark", "dataset", "delta", "evaluated_at", "model_id", "model_revision", "name", "owner",
        "source", "steps", "tasks", "user")


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_ledger(path: Path) -> list[dict]:
    """Read and verify the hash chain; any edited or reordered line fails loudly."""
    if not path.exists():
        return []
    entries, prev = [], GENESIS
    for n, line in enumerate(path.read_text().splitlines(), 1):
        entry = json.loads(line)
        if entry.get("prev_sha256") != prev or entry.get("seq") != n:
            raise SystemExit(f"{path}:{n}: ledger chain broken; it is append-only and must not be edited")
        prev = hashlib.sha256(line.encode()).hexdigest()
        entries.append(entry)
    return entries


def parse_result(run: dict) -> tuple[str, dict]:
    """Validate one leaderboard ``runs[]`` record: run_id, submission_id, user, scores.<domain>.{solved,
    total, score}, solved, total. Domain ids must be the arena's; totals must add up."""
    missing = [k for k in RUN_KEYS if k not in run]
    if missing:
        raise SystemExit(f"not a leaderboard run record; missing {missing}")
    if not run["run_id"] or not isinstance(run["scores"], dict) or not run["scores"]:
        raise SystemExit("run record needs a run_id and non-empty scores")
    clean = {}
    for name, value in run["scores"].items():
        if name not in DOMAIN_IDS:
            raise SystemExit(f"unknown domain id {name!r}")
        if not isinstance(value, dict) or not {"solved", "total"} <= set(value):
            raise SystemExit(f"domain {name!r} needs solved and total")
        solved, total = value["solved"], value["total"]
        if type(solved) is not int or type(total) is not int or not 0 <= solved <= total or total < 1:
            raise SystemExit(f"domain {name!r}: need integers 0 <= solved <= total, total >= 1")
        clean[name] = {"solved": solved, "total": total}
    for key in ("solved", "total"):
        if run[key] != sum(d[key] for d in clean.values()):
            raise SystemExit(f"run {key} {run[key]} disagrees with the per-domain sum")
    return str(run["run_id"]), dict(sorted(clean.items()))


def load_receipt(receipt_path: Path, request_path: Path | None = None) -> tuple[dict, dict]:
    """Return (receipt, request) after checking the request hashes to the receipt's request_sha256."""
    receipt = json.loads(receipt_path.read_text())
    if request_path is None:
        request_path = receipt_path.with_name(receipt_path.name.replace("receipt", "request", 1))
    request = json.loads(request_path.read_text())
    sha = canonical_sha(request)
    if sha != receipt.get("request_sha256"):
        raise SystemExit(f"{request_path} hashes to {sha}, not the receipt's {receipt.get('request_sha256')}")
    if request.get("submission_id") != receipt.get("submission_id"):
        raise SystemExit("request and receipt name different submission ids")
    return receipt, request


def append_run(ledger: Path, run: dict, receipt: dict, request: dict, now: str | None = None) -> dict:
    run_id, domains = parse_result(run)
    if run["submission_id"] != receipt["submission_id"]:
        raise SystemExit(f"run is for {run['submission_id']}, receipt is for {receipt['submission_id']}")
    entries = read_ledger(ledger)
    if any(e["run_id"] == run_id for e in entries):
        raise SystemExit(f"ledger already holds run {run_id}; entries are never replaced")
    if any(e["submission_id"] == run["submission_id"] for e in entries):
        print(f"note: {run['submission_id']} already has a run in the ledger; this is another run of it",
              file=sys.stderr)
    prev = hashlib.sha256(ledger.read_text().splitlines()[-1].encode()).hexdigest() if entries else GENESIS
    entry = {"seq": len(entries) + 1, "prev_sha256": prev,
             "recorded_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "submission_id": run["submission_id"], "request_sha256": receipt["request_sha256"],
             "run_id": run_id, "cells": [t["task_id"] for t in request.get("tasks", [])], "domains": domains,
             "solved": run["solved"], "total": run["total"], **{k: run[k] for k in KEPT if k in run}}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return entry


def ledger_add(ledger: Path, receipt_path: Path, result_path: Path, request_path: Path | None = None,
               now: str | None = None) -> dict:
    receipt, request = load_receipt(receipt_path, request_path)
    return append_run(ledger, json.loads(result_path.read_text()), receipt, request, now)


def http_get_json(url: str, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "qe-flywheel/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_leaderboard(get=http_get_json, url: str = LEADERBOARD, max_pages: int = 50) -> dict:
    """GET the leaderboard and follow ``next_cursor`` pages, merging their runs."""
    board = get(url)
    if not isinstance(board.get("runs"), list) or "scores" not in board.get("baseline", {}):
        raise SystemExit("leaderboard response has no runs list or baseline scores")
    cursor, pages = board.get("next_cursor"), 1
    while cursor:
        if not re.fullmatch(r"[0-9]+-[0-9]+", str(cursor)) or pages >= max_pages:
            raise SystemExit(f"refusing leaderboard cursor {cursor!r} at page {pages}")
        page = get(f"{url}/{cursor}")
        board["runs"] += page.get("runs", [])
        cursor, pages = page.get("next_cursor"), pages + 1
    return board


def baseline_from(board: dict) -> dict[str, int]:
    return {name: d["solved"] for name, d in board["baseline"]["scores"].items() if d["solved"]}


def ledger_fetch(ledger: Path, user: str, submission_id: str | None = None, receipts: Path | None = None,
                 board: dict | None = None, get=http_get_json) -> dict:
    """Append this user's leaderboard runs that are not in the ledger yet (idempotent by run_id). A run
    is appended only with its local receipt, so every entry carries a verified request sha. The
    leaderboard baseline is saved beside the ledger for ``show``."""
    board = board if board is not None else fetch_leaderboard(get)
    receipts = receipts or ledger.parent
    by_submission = {}
    for path in sorted(receipts.glob("receipt-*.json")):
        sid = json.loads(path.read_text()).get("submission_id")
        if sid:
            by_submission[sid] = path
    have = {e["run_id"] for e in read_ledger(ledger)}
    report = {"appended": [], "already": [], "no_receipt": []}
    runs = [r for r in board["runs"] if r.get("user") == user
            and (submission_id is None or r.get("submission_id") == submission_id)]
    for run in sorted(runs, key=lambda r: (r.get("evaluated_at") or 0, r.get("run_id"))):
        if run.get("run_id") in have:
            report["already"].append(run["run_id"])
        elif run.get("submission_id") not in by_submission:
            report["no_receipt"].append(f"{run.get('submission_id')}/{run.get('run_id')}")
        else:
            receipt, request = load_receipt(by_submission[run["submission_id"]])
            report["appended"].append(append_run(ledger, run, receipt, request)["run_id"])
            have.add(run["run_id"])
    base = {"source": LEADERBOARD, "baseline": board["baseline"], "solved_by_domain": baseline_from(board)}
    sidecar = baseline_path(ledger)
    if not sidecar.exists() or json.loads(sidecar.read_text()) != base:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    report["matched"] = len(runs)
    return report


def baseline_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.stem + "-baseline.json")


def resolve_baseline(ledger: Path, explicit: str | None) -> tuple[dict[str, int], str]:
    """--baseline wins; then the leaderboard baseline saved by the last fetch; then the offline fallback."""
    if explicit:
        return parse_baseline(explicit), "--baseline"
    sidecar = baseline_path(ledger)
    if sidecar.exists():
        return json.loads(sidecar.read_text())["solved_by_domain"], f"leaderboard baseline saved in {sidecar.name}"
    return parse_baseline(BASELINE), "OFFLINE FALLBACK (run `ledger fetch` to read the leaderboard's baseline)"


def parse_baseline(text: str) -> dict[str, int]:
    """``name=solved,...`` or a saved leaderboard JSON, whose ``baseline.scores`` is used."""
    if Path(text).is_file():
        return baseline_from(json.loads(Path(text).read_text()))
    out = {}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        out[name.strip()] = int(value)
    return out


def binom_pmf(n: int, p: float) -> np.ndarray:
    return np.array([math.comb(n, x) * p ** x * (1.0 - p) ** (n - x) for x in range(n + 1)])


def max_pmf(n: int, p: float, k: int) -> np.ndarray:
    """Exact pmf of the max of k iid Binomial(n, p): P(max <= x) = F(x)^k."""
    cdf = np.cumsum(binom_pmf(n, p)) ** k
    return np.diff(np.concatenate([[0.0], cdf]))


def null_composite(k: int, domains: dict[str, tuple[int, float]]) -> np.ndarray:
    """Exact distribution of the best-per-domain composite of k runs when every run is the baseline."""
    dist = np.array([1.0])
    for n, p in domains.values():
        dist = np.convolve(dist, max_pmf(n, p, k))
    return dist


def null_single(domains: dict[str, tuple[int, float]]) -> np.ndarray:
    dist = np.array([1.0])
    for n, p in domains.values():
        dist = np.convolve(dist, binom_pmf(n, p))
    return dist


def tail(dist: np.ndarray, x: int) -> float:
    return float(dist[x:].sum()) if x < len(dist) else 0.0


def null_domains(entries: list[dict], baseline: dict[str, int]) -> dict[str, tuple[int, float]]:
    totals = {}
    for e in entries:
        for name, d in e["domains"].items():
            if totals.setdefault(name, d["total"]) != d["total"]:
                raise SystemExit(f"domain {name!r} has different totals across runs")
    for name, solved in baseline.items():
        if name not in totals:
            raise SystemExit(f"baseline domain {name!r} never appears in a result")
        if not 0 <= solved <= totals[name]:
            raise SystemExit(f"baseline {name}={solved} outside 0..{totals[name]}")
    return {name: (n, baseline.get(name, 0) / n) for name, n in sorted(totals.items())}


def ledger_show(entries: list[dict], baseline: dict[str, int], out=None, source: str = "") -> dict:
    out = out or sys.stdout
    if not entries:
        print("ledger is empty", file=out)
        return {}
    nulls = null_domains(entries, baseline)
    single = null_single(nulls)
    names = list(nulls)
    print("PER-RUN SCORES: each row is one noisy single attempt on the held-out set", file=out)
    print(f"{'seq':>3}  {'submission':<22}{'run':<18}{'total':>8}  " + "".join(f"{n[:10]:>11}" for n in names)
          + "   P(null run >= this)", file=out)
    for e in entries:
        cols = "".join(f"{e['domains'].get(n, {}).get('solved', '-'):>11}" for n in names)
        print(f"{e['seq']:>3}  {e['submission_id']:<22}{e['run_id']:<18}{e['solved']:>4}/{e['total']:<3}  {cols}"
              f"   {tail(single, e['solved']):.3f}", file=out)
    best = {n: max(e["domains"].get(n, {}).get("solved", 0) for e in entries) for n in names}
    composite = sum(best.values())
    k = len(entries)
    dist = null_composite(k, nulls)
    expected = float(np.dot(np.arange(len(dist)), dist))
    baseline_total = sum(baseline.values())
    print(f"\nBEST-PER-DOMAIN COMPOSITE (not a run score; selection-inflated): {composite}  "
          + " + ".join(f"{n} {v}" for n, v in best.items()), file=out)
    print(f"null: if all {k} runs were the baseline ({baseline_total} = "
          + " + ".join(f"{n} {v}" for n, v in baseline.items())
          + f"), the composite is {expected:.2f} in expectation (exact), "
          f"P(null composite >= {composite}) = {tail(dist, composite):.3f}", file=out)
    if source:
        print(f"baseline source: {source}", file=out)
    return {"composite": composite, "best": best, "null_expected": expected,
            "null_p_value": tail(dist, composite), "runs": k}
