"""Append-only, hash-chained ledger of arena runs, and how to read a composite against its null.

A result is one ``runs[]`` record from the public leaderboard (GET /api/leaderboard, no auth). Every
entry keeps the record it was built from and is re-validated on every read; the chain head is anchored
in a ``.head`` sidecar, so editing, reordering or deleting any line, the last included, fails."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import null

GENESIS = "0" * 64
LEADERBOARD = "https://openenvarena-arena.hf.space/api/leaderboard"
MAX_BYTES = 8 * 1024 * 1024
DOMAIN_IDS = ("software-engineering", "industrial-physical-systems", "natural-science", "office-white-collar",
              "finance-economics", "mathematics-or-formal-reasoning", "media-content-production", "cybersecurity")
RUN_KEYS = ("run_id", "submission_id", "user", "scores", "solved", "total", "benchmark")
TASKS_PER_DOMAIN = 5
# Offline fallback only: the leaderboard's own `baseline` (Qwen3.8-27B 1d4bf0f, measured 2026-10-07) wins.
BASELINE = "industrial-physical-systems=2,office-white-collar=2"


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def line_sha(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def head_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.name + ".head")


def baselines_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.stem + "-baselines.jsonl")


def validate_scores(scores, what: str) -> dict:
    """All 8 domains, integer solved/total with 0 <= solved <= total and total >= 1."""
    if not isinstance(scores, dict) or set(scores) != set(DOMAIN_IDS):
        got = sorted(scores) if isinstance(scores, dict) else type(scores).__name__
        raise SystemExit(f"{what}: scores must cover exactly the 8 arena domains, got {got}")
    clean = {}
    for name in DOMAIN_IDS:
        value = scores[name]
        if not isinstance(value, dict) or not {"solved", "total"} <= set(value):
            raise SystemExit(f"{what}: domain {name!r} needs solved and total")
        solved, total = value["solved"], value["total"]
        if type(solved) is not int or type(total) is not int or not 0 <= solved <= total or total < 1:
            raise SystemExit(f"{what}: domain {name!r} needs integers 0 <= solved <= total, total >= 1")
        clean[name] = {"solved": solved, "total": total}
    return clean


def parse_result(run: dict) -> tuple[str, dict]:
    """Validate one leaderboard ``runs[]`` record; returns (run_id, domains)."""
    if not isinstance(run, dict):
        raise SystemExit("a run record must be an object")
    missing = [k for k in RUN_KEYS if k not in run]
    if missing:
        raise SystemExit(f"not a leaderboard run record; missing {missing}")
    if not run["run_id"] or not isinstance(run["run_id"], str):
        raise SystemExit("run record needs a run_id")
    clean = validate_scores(run["scores"], f"run {run['run_id']}")
    for key in ("solved", "total"):
        if run[key] != sum(d[key] for d in clean.values()):
            raise SystemExit(f"run {key} {run[key]} disagrees with the per-domain sum")
    return run["run_id"], clean


def read_ledger(path: Path) -> list[dict]:
    """Read and verify: chain, sequence, head anchor, and every entry against its own record."""
    head = head_path(path)
    if not path.exists():
        if head.exists():
            raise SystemExit(f"{path} is missing but {head.name} anchors a head")
        return []
    text = path.read_text()
    if text and not text.endswith("\n"):
        raise SystemExit(f"{path}: last line is not terminated; the ledger was truncated or edited")
    entries, prev, last = [], GENESIS, None
    for n, line in enumerate(text.splitlines(), 1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(f"{path}:{n}: not a ledger entry (blank or corrupt line); chain broken") from None
        if not isinstance(entry, dict) or entry.get("prev_sha256") != prev or entry.get("seq") != n:
            raise SystemExit(f"{path}:{n}: ledger chain broken; it is append-only and must not be edited")
        run_id, domains = parse_result(entry.get("record"))
        record = entry["record"]
        derived = {"run_id": run_id, "domains": domains, "solved": record["solved"], "total": record["total"],
                   "submission_id": record["submission_id"], "benchmark": record["benchmark"]}
        if any(entry.get(k) != v for k, v in derived.items()):
            raise SystemExit(f"{path}:{n}: entry disagrees with its own run record")
        prev = last = line_sha(line)
        entries.append(entry)
    if entries:
        anchor = json.loads(head.read_text()) if head.exists() else None
        if anchor != {"seq": len(entries), "sha256": last}:
            raise SystemExit(f"{path}: head anchor {anchor} does not match the last entry "
                             f"(seq {len(entries)}, {last[:12]}); a line was removed or edited")
    elif head.exists():
        raise SystemExit(f"{path} is empty but {head.name} anchors a head")
    return entries


def load_receipt(receipt_path: Path, request_path: Path | None = None) -> tuple[dict, dict]:
    """(receipt, request) after checking the receipt says the POST happened and the request hashes."""
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("state") != "recorded" or receipt.get("post_attempted") is not True:
        raise SystemExit(f"{receipt_path}: receipt state {receipt.get('state')!r}, post_attempted "
                         f"{receipt.get('post_attempted')!r}; only a recorded submission can enter the ledger")
    if request_path is None:
        request_path = receipt_path.with_name(receipt_path.name.replace("receipt", "request", 1))
    request = json.loads(request_path.read_text())
    sha = canonical_sha(request)
    if sha != receipt.get("request_sha256"):
        raise SystemExit(f"{request_path} hashes to {sha}, not the receipt's {receipt.get('request_sha256')}")
    if request.get("submission_id") != receipt.get("submission_id"):
        raise SystemExit("request and receipt name different submission ids")
    return receipt, request


def append_run(ledger: Path, run: dict, receipt: dict, request: dict, source: str, now: str | None = None) -> dict:
    if source not in ("fetched", "manual"):
        raise ValueError("source is fetched or manual")
    run_id, domains = parse_result(run)
    if run["submission_id"] != receipt["submission_id"]:
        raise SystemExit(f"run is for {run['submission_id']}, receipt is for {receipt['submission_id']}")
    entries = read_ledger(ledger)
    if any(e["run_id"] == run_id for e in entries):
        raise SystemExit(f"ledger already holds run {run_id}; entries are never replaced")
    prev = line_sha(ledger.read_text().splitlines()[-1]) if entries else GENESIS
    entry = {"seq": len(entries) + 1, "prev_sha256": prev, "source": source,
             "recorded_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "submission_id": run["submission_id"], "request_sha256": receipt["request_sha256"],
             "run_id": run_id, "cells": [t["task_id"] for t in request.get("tasks", [])], "domains": domains,
             "solved": run["solved"], "total": run["total"], "benchmark": run["benchmark"], "record": run}
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(line + "\n")
    head_path(ledger).write_text(json.dumps({"seq": entry["seq"], "sha256": line_sha(line)}) + "\n")
    return entry


def ledger_add(ledger: Path, receipt_path: Path, result_path: Path, request_path: Path | None = None,
               now: str | None = None) -> dict:
    """A hand-entered record. It is kept apart: ``show`` ignores manual entries unless asked."""
    receipt, request = load_receipt(receipt_path, request_path)
    return append_run(ledger, json.loads(result_path.read_text()), receipt, request, "manual", now)


def http_get_json(url: str, timeout: float = 30.0, max_bytes: int = MAX_BYTES) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "qe-flywheel/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise SystemExit(f"leaderboard response is over {max_bytes} bytes; refusing it")
    return json.loads(body)


def check_board(board: dict) -> dict:
    if not isinstance(board, dict) or not isinstance(board.get("runs"), list):
        raise SystemExit("leaderboard response has no runs list")
    validate_scores((board.get("baseline") or {}).get("scores"), "leaderboard baseline")
    return board


def fetch_leaderboard(get=http_get_json, url: str = LEADERBOARD, cursor_url: str | None = None,
                      max_pages: int = 50) -> dict:
    """GET the leaderboard. The cursor format is unverified: on 2026-10-12 the response had
    next_cursor null, the Space's own page reads /api/leaderboard once and never sends a cursor,
    `{url}/1-1` returned 404 and `?cursor=1-1` returned the unpaged board. So a non-null next_cursor
    is refused unless the caller supplies ``cursor_url`` (e.g. "{url}?cursor={cursor}")."""
    board = check_board(get(url))
    cursor, seen = board.get("next_cursor"), set()
    while cursor:
        if cursor_url is None:
            raise SystemExit(f"leaderboard returned next_cursor {cursor!r} but its pagination format is "
                             "unverified; refusing a partial board (pass --cursor-url once it is known)")
        if not re.fullmatch(r"[0-9]+-[0-9]+", str(cursor)):
            raise SystemExit(f"refusing leaderboard cursor {cursor!r}")
        if cursor in seen:
            raise SystemExit(f"leaderboard repeated cursor {cursor!r}; stopping")
        if len(seen) + 1 >= max_pages:
            raise SystemExit(f"leaderboard still paging after {max_pages} pages; stopping")
        seen.add(cursor)
        page = get(cursor_url.format(url=url, cursor=cursor))
        if not isinstance(page, dict) or not isinstance(page.get("runs"), list):
            raise SystemExit(f"leaderboard page {cursor!r} has no runs list")
        board["runs"] += page["runs"]
        cursor = page.get("next_cursor")
    return board


def save_baseline(ledger: Path, board: dict, now: str | None = None) -> dict:
    """Append the leaderboard baseline to a versioned JSONL when it differs from the last one kept."""
    scores = validate_scores(board["baseline"]["scores"], "leaderboard baseline")
    path = baselines_path(ledger)
    history = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    record = {"fetched_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": LEADERBOARD,
              "benchmark": board.get("benchmark"), "measured": board["baseline"].get("measured"), "scores": scores}
    if not history or {k: history[-1].get(k) for k in ("benchmark", "scores")} != \
            {k: record[k] for k in ("benchmark", "scores")}:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record
    return history[-1]


def ledger_fetch(ledger: Path, user: str, submission_id: str | None = None, receipts: Path | None = None,
                 board: dict | None = None, get=http_get_json, cursor_url: str | None = None) -> dict:
    """Append this user's leaderboard runs that are not in the ledger yet (idempotent by run_id). A run
    is appended only with its local, recorded receipt. ``user`` matches exactly; a case-only match is
    reported, not taken."""
    board = check_board(board) if board is not None else fetch_leaderboard(get, cursor_url=cursor_url)
    receipts = receipts or ledger.parent
    by_submission = {}
    for path in sorted(receipts.glob("receipt-*.json")):
        sid = json.loads(path.read_text()).get("submission_id")
        if sid:
            by_submission[sid] = path
    have = {e["run_id"] for e in read_ledger(ledger)}
    report = {"appended": [], "already": [], "no_receipt": [], "case_mismatch": []}
    runs = []
    for run in board["runs"]:
        if submission_id is not None and run.get("submission_id") != submission_id:
            continue
        if run.get("user") == user:
            runs.append(run)
        elif str(run.get("user", "")).lower() == user.lower():
            report["case_mismatch"].append(run.get("user"))
    if report["case_mismatch"]:
        print(f"warning: no exact match for user {user!r}, but runs exist for "
              f"{sorted(set(report['case_mismatch']))}; users match case-sensitively", file=sys.stderr)
    for run in sorted(runs, key=lambda r: (r.get("evaluated_at") or 0, str(r.get("run_id")))):
        if run.get("run_id") in have:
            report["already"].append(run["run_id"])
        elif run.get("submission_id") not in by_submission:
            report["no_receipt"].append(f"{run.get('submission_id')}/{run.get('run_id')}")
        else:
            receipt, request = load_receipt(by_submission[run["submission_id"]])
            report["appended"].append(append_run(ledger, run, receipt, request, "fetched")["run_id"])
            have.add(run["run_id"])
    report["baseline"] = save_baseline(ledger, board)["fetched_at"]
    report["matched"] = len(runs)
    return report


def parse_baseline(text: str) -> dict:
    """``name=solved,...`` (unnamed domains 0 of 5) or a saved leaderboard JSON's ``baseline.scores``."""
    if Path(text).is_file():
        return validate_scores(json.loads(Path(text).read_text())["baseline"]["scores"], text)
    scores = {d: {"solved": 0, "total": TASKS_PER_DOMAIN} for d in DOMAIN_IDS}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        if name.strip() not in scores:
            raise SystemExit(f"unknown baseline domain {name.strip()!r}")
        scores[name.strip()]["solved"] = int(value)
    return validate_scores(scores, "--baseline")


def resolve_baseline(ledger: Path, explicit: str | None) -> tuple[dict, str]:
    """--baseline wins; then the latest leaderboard baseline kept by fetch; then the offline fallback."""
    if explicit:
        return parse_baseline(explicit), "--baseline"
    path = baselines_path(ledger)
    if path.exists():
        latest = json.loads(path.read_text().splitlines()[-1])
        return validate_scores(latest["scores"], path.name), \
            f"leaderboard baseline fetched {latest['fetched_at']} ({latest.get('benchmark')})"
    return parse_baseline(BASELINE), "OFFLINE FALLBACK (run `ledger fetch` to read the leaderboard's baseline)"


def null_domains(entries: list[dict], baseline: dict) -> dict[str, tuple[int, int]]:
    """(tasks, baseline solved) per domain, after checking the baseline and every run agree on totals."""
    for e in entries:
        for name, d in e["domains"].items():
            if d["total"] != baseline[name]["total"]:
                raise SystemExit(f"domain {name!r}: run {e['run_id']} has {d['total']} tasks, "
                                 f"the baseline {baseline[name]['total']}")
    return {name: (baseline[name]["total"], baseline[name]["solved"]) for name in DOMAIN_IDS}


def ledger_show(entries: list[dict], baseline: dict, out=None, source: str = "", include_manual: bool = False,
                head: dict | None = None) -> dict:
    out = out or sys.stdout
    if head:
        print(f"ledger head: seq {head['seq']} sha256 {head['sha256']}", file=out)
    manual = [e for e in entries if e.get("source") != "fetched"]
    if manual and not include_manual:
        print(f"{len(manual)} manual entr{'y' if len(manual) == 1 else 'ies'} left out (--include-manual)", file=out)
        entries = [e for e in entries if e.get("source") == "fetched"]
    if not entries:
        print("no runs to show", file=out)
        return {}
    benchmarks = sorted({e["benchmark"] for e in entries})
    if len(benchmarks) > 1:
        raise SystemExit(f"runs span benchmarks {benchmarks}; a composite across them means nothing")
    domains = null_domains(entries, baseline)
    single = null.composite_pmf(1, domains)
    print(f"PER-RUN SCORES ({benchmarks[0]}): each row is one noisy single attempt on the held-out set", file=out)
    print(f"{'seq':>3}  {'submission':<28}{'run':<26}{'total':>7}  P(model null run >= this)", file=out)
    for e in entries:
        print(f"{e['seq']:>3}  {e['submission_id']:<28}{e['run_id']:<26}{e['solved']:>3}/{e['total']:<3}  "
              f"{null.tail(single, e['solved']):.3f}", file=out)
    best = {n: max(e["domains"][n]["solved"] for e in entries) for n in DOMAIN_IDS}
    composite, k = sum(best.values()), len(entries)
    model, plugin = null.composite_pmf(k, domains), null.composite_pmf(k, domains, "plugin")
    print(f"\nBEST-PER-DOMAIN COMPOSITE (not a run score; selection-inflated): {composite}  "
          + " + ".join(f"{n} {v}" for n, v in best.items() if v), file=out)
    print(f"model-based null, {k} runs no better than the baseline ({sum(d[1] for d in domains.values())}/"
          f"{sum(d[0] for d in domains.values())}, Jeffreys beta-binomial per domain): E = {null.mean(model):.2f}, "
          f"P(null >= {composite}) = {null.tail(model, composite):.3f}", file=out)
    sens = {z: null.composite_pmf(k, domains, p_zero=z) for z in (0.0, 0.02, 0.05)}
    print("sensitivity, rate of the baseline's 0/5 domains fixed at p_zero: "
          + "; ".join(f"p_zero={z}: E {null.mean(d):.2f}, P(>= {composite}) {null.tail(d, composite):.3f}"
                      for z, d in sens.items()), file=out)
    print(f"lower reference, plug-in binomial at s/n: E = {null.mean(plugin):.2f}, "
          f"P(>= {composite}) = {null.tail(plugin, composite):.3f}", file=out)
    if source:
        print(f"baseline source: {source}", file=out)
    return {"composite": composite, "best": best, "runs": k, "null_expected": null.mean(model),
            "null_p_value": null.tail(model, composite), "plugin_expected": null.mean(plugin),
            "sensitivity": {str(z): null.tail(d, composite) for z, d in sens.items()},
            "per_run_p": {e["run_id"]: null.tail(single, e["solved"]) for e in entries}}
