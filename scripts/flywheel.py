#!/usr/bin/env python3
"""Daily flywheel: pick task cells that give GRPO a gradient, and keep honest score books.

GRPO learns only from groups whose rewards are not all equal, so we want cells the model
sometimes solves. One arena run is noise and best-per-domain composites inflate, so every
decision here is paired and pre-registered, and every composite is shown next to its null.

    flywheel.py stats RUNS.jsonl... [--json OUT]
    flywheel.py select RUNS.jsonl... --max-cells 6 --seed 1 --template submission/tasks-v1.json --out tasks-v2.json
    flywheel.py ledger fetch --user Profa-QE [--submission-id ID]
    flywheel.py ledger add --receipt submission/receipt-v1.json --result run.json
    flywheel.py ledger show
    flywheel.py onecell-search init --template submission/tasks-v1.json --out knobs.json
    flywheel.py onecell-search propose --config knobs.json --seed 1 --runs RUNS.jsonl... --out plan.json
    flywheel.py onecell-search decide --plan plan.json --baseline-discovery A --candidate-discovery B \\
        --baseline-confirm C --candidate-confirm D

Inputs are the JSONL files written by scripts/calibrate.py. Rows with ``reward: null`` are
harness errors: they are counted and excluded, never scored as 0. Stdlib and numpy only; the
logic lives in scripts/qe_flywheel/ (kept out of qe_arena/, which ships in the image).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from qe_flywheel import ledger, report, search, select, stats

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats", help="per-cell posteriors from calibration runs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--json", type=Path)
    p = sub.add_parser("select", help="Thompson-sample a task list")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--max-cells", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--why", type=Path, help="sidecar; default <out>.why.json")
    p.add_argument("--min-mixed", type=float, default=select.MIN_MIXED,
                   help="exclude cells whose posterior-mean p_mix@G is below this (rarely mixed)")
    p.add_argument("--min-groups", type=int, default=1, help="same-seed groups needed (group objective only)")
    p.add_argument("--min-episodes", type=int, default=4, help="valid episodes needed")
    p.add_argument("--objective", choices=sorted(select.OBJECTIVES), default="projected",
                   help="; ".join(f"{k}: {v}" for k, v in select.OBJECTIVES.items()))
    p.add_argument("--group-size", type=int, default=stats.TRAIN_GROUP,
                   help="G for the projected objective (arena trains on 4 of 5 episodes)")
    p = sub.add_parser("ledger", help="append-only ledger of arena runs")
    lsub = p.add_subparsers(dest="action", required=True)
    for name in ("fetch", "add", "show"):
        q = lsub.add_parser(name)
        q.add_argument("--ledger", type=Path, default=SUBMISSION / "ledger.jsonl")
        if name == "fetch":
            q.add_argument("--user", required=True, help="leaderboard user, e.g. Profa-QE")
            q.add_argument("--submission-id")
            q.add_argument("--receipts", type=Path, default=SUBMISSION, help="where receipt-vN.json files live")
            q.add_argument("--from-file", type=Path, help="a saved leaderboard response instead of the network")
            q.add_argument("--timeout", type=float, default=30.0)
            q.add_argument("--cursor-url", help="pagination URL template, e.g. '{url}?cursor={cursor}'; unverified, "
                                                 "so a paged response is refused without it")
        elif name == "add":
            q.add_argument("--receipt", type=Path, required=True)
            q.add_argument("--result", type=Path, required=True, help="one leaderboard runs[] record")
            q.add_argument("--request", type=Path, help="default: the receipt's sibling request-vN.json")
        else:
            q.add_argument("--baseline", help="name=solved,... or a saved leaderboard JSON; default: last fetch")
            q.add_argument("--include-manual", action="store_true", help="also show hand-entered runs")
            q.add_argument("--json", type=Path)
    p = sub.add_parser("onecell-search", help="Darwin-style one-knob proposal and paired decision")
    osub = p.add_subparsers(dest="action", required=True)
    q = osub.add_parser("init")
    q.add_argument("--template", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q = osub.add_parser("propose")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--seed", type=int, required=True)
    q.add_argument("--runs", nargs="*", type=Path, default=[], help="seeds seen here are never confirmation seeds")
    q.add_argument("--history", type=Path, help="JSONL of past decisions; their moves are not re-proposed")
    q.add_argument("--discovery-seeds", type=int, default=32)
    q.add_argument("--confirm-seeds", type=int, default=32)
    q.add_argument("--attempts", type=int, default=1)
    q.add_argument("--margin", type=float, default=0.0, help="discovery bootstrap lower bound must exceed this")
    q.add_argument("--confirm-tolerance", type=float, default=0.05,
                   help="confirmation bootstrap lower bound must exceed minus this")
    q.add_argument("--error-tolerance", type=float, default=0.05, help="max gap in error/budget rate between arms")
    q.add_argument("--metric", choices=search.METRICS, default="mixed_rate",
                   help="mixed_rate (default): P(G fresh instances not all equal), the trainer's gradient signal")
    q.add_argument("--out", type=Path, required=True)
    q = osub.add_parser("decide")
    q.add_argument("--plan", type=Path, required=True)
    for name in ("baseline-discovery", "candidate-discovery", "baseline-confirm", "candidate-confirm"):
        q.add_argument(f"--{name}", type=Path, required=True)
    q.add_argument("--out", type=Path)
    return parser


def run_select(args) -> int:
    cell_stats = stats.compute_stats(stats.load_runs(args.runs))
    template = json.loads(args.template.read_text())
    tasks, reasons = select.select_cells(cell_stats, template, args.max_cells, args.seed, args.min_mixed,
                                         args.min_groups, args.min_episodes, args.objective, args.group_size)
    write_json(args.out, tasks)
    why = args.why or args.out.with_suffix(".why.json")
    write_json(why, {"seed": args.seed, "max_cells": args.max_cells, "objective": args.objective,
                     "measure": select.OBJECTIVES[args.objective], "group_size": args.group_size,
                     "min_mixed": args.min_mixed, "min_groups": args.min_groups, "min_episodes": args.min_episodes,
                     "template": {"path": str(args.template), "sha256": file_sha(args.template)},
                     "runs": [{"path": str(r), "sha256": file_sha(r)} for r in args.runs], "cells": reasons})
    print(f"objective {args.objective}: {select.OBJECTIVES[args.objective]} (G={args.group_size})")
    for r in reasons:
        print(f"{r['decision']:<9} {r['task_id']:<22} {r['reason']}")
    print(f"\n{len(tasks)} cells -> {args.out}; reasons -> {why}")
    if not tasks:
        print("no cell qualified; nothing to submit", file=sys.stderr)
        return 4
    return 0


def run_ledger(args) -> int:
    if args.action == "fetch":
        board = json.loads(args.from_file.read_text()) if args.from_file else ledger.fetch_leaderboard(
            lambda url: ledger.http_get_json(url, args.timeout), cursor_url=args.cursor_url)
        fetched = ledger.ledger_fetch(args.ledger, args.user, args.submission_id, args.receipts, board=board)
        print(json.dumps(fetched, indent=2))
    elif args.action == "add":
        print(json.dumps(ledger.ledger_add(args.ledger, args.receipt, args.result, args.request), indent=2))
    else:
        baseline, source = ledger.resolve_baseline(args.ledger, args.baseline)
        entries = ledger.read_ledger(args.ledger)
        head = json.loads(ledger.head_path(args.ledger).read_text()) if entries else None
        summary = report.ledger_show(entries, baseline, ledger.load_population(args.ledger), source=source,
                                     include_manual=args.include_manual, head=head)
        if args.json:
            write_json(args.json, summary)
    return 0


def run_search(args) -> int:
    if args.action == "init":
        write_json(args.out, search.init_config(json.loads(args.template.read_text())))
        print(args.out.read_text())
        return 0
    if args.action == "propose":
        config = json.loads(args.config.read_text())
        seen = {r["seed"] for r in stats.load_runs(args.runs)} if args.runs else set()
        history = [json.loads(line) for line in args.history.read_text().splitlines() if line.strip()] \
            if args.history and args.history.exists() else []
        plan = search.propose(config, args.seed, seen, history, sorted(config["completion_tokens"]),
                              args.discovery_seeds, args.confirm_seeds, args.attempts, args.margin,
                              args.confirm_tolerance, args.metric, error_tolerance=args.error_tolerance)
        write_json(args.out, plan)
        print(f"knob {plan['knob']}: {plan['from']} -> {plan['to']}; discovery seeds "
              f"{plan['discovery']['seed_base']}+{len(plan['discovery']['seeds'])}, confirmation "
              f"{plan['confirmation']['seed_base']}+{len(plan['confirmation']['seeds'])} -> {args.out}")
        return 0
    plan = json.loads(args.plan.read_text())
    result = search.decide(plan, *(stats.load_runs([path]) for path in (
        args.baseline_discovery, args.candidate_discovery, args.baseline_confirm, args.candidate_confirm)))
    print(json.dumps(result, indent=2))
    if args.out:
        write_json(args.out, result)
    return 0 if result["accept"] else 3


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "stats":
        cell_stats = stats.compute_stats(stats.load_runs(args.runs))
        stats.print_stats(cell_stats)
        if args.json:
            write_json(args.json, cell_stats)
        return 0
    if args.cmd == "select":
        return run_select(args)
    if args.cmd == "ledger":
        return run_ledger(args)
    return run_search(args)


if __name__ == "__main__":
    sys.exit(main())
