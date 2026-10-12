#!/usr/bin/env python3
"""Daily flywheel: pick task cells that give GRPO a gradient, and keep honest score books.

GRPO learns only from groups whose rewards are not all equal, so we want cells the model
sometimes solves. One arena run is noise and best-per-domain composites inflate, so every
decision here is paired and pre-registered, and every composite is shown next to its null.

    flywheel.py stats RUNS.jsonl... [--json OUT]
    flywheel.py select RUNS.jsonl... --max-cells 6 --seed 1 --template submission/tasks-v1.json --out tasks-v2.json
    flywheel.py ledger add --receipt submission/receipt-v1.json --result result.json
    flywheel.py ledger show
    flywheel.py onecell-search init --template submission/tasks-v1.json --out knobs.json
    flywheel.py onecell-search propose --config knobs.json --seed 1 --runs RUNS.jsonl... --out plan.json
    flywheel.py onecell-search decide --plan plan.json --baseline-discovery A --candidate-discovery B \
        --baseline-confirm C --candidate-confirm D

Inputs are the JSONL files written by scripts/calibrate.py. Rows with ``reward: null`` are
harness errors: they are counted and excluded, never scored as 0. Stdlib and numpy only.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("d1", "d2", "d3")
PRIOR = 0.5  # Jeffreys: Beta(0.5, 0.5)
CI = 0.90
ARENA_GROUP = 5  # DESIGN.md: each optimizer step plays 5 episodes of one task, unseeded


# ---------------------------------------------------------------- Beta posterior math

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if beta_cdf(mid, a, b) < q else (lo, mid)
    return (lo + hi) / 2.0


def posterior(k: int, n: int) -> dict:
    """Jeffreys posterior for k successes in n trials, with the equal-tailed 90% interval."""
    a, b = k + PRIOR, n - k + PRIOR
    tail = (1.0 - CI) / 2.0
    return {"k": k, "n": n, "a": a, "b": b, "mean": a / (a + b),
            "lo": beta_ppf(tail, a, b), "hi": beta_ppf(1.0 - tail, a, b)}


def projected_informative(a: float, b: float, g: int) -> float:
    """E[1 - p^g - (1-p)^g] for p ~ Beta(a, b): the chance that g independent episodes
    are not all solved and not all unsolved. Binary approximation of 'not all equal'."""
    all_solved = math.prod((a + i) / (a + b + i) for i in range(g))
    none_solved = math.prod((b + i) / (a + b + i) for i in range(g))
    return 1.0 - all_solved - none_solved


# ---------------------------------------------------------------- loading and grouping

def load_runs(paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row.pop("messages", None)
                for key in ("task_id", "seed", "attempt"):
                    if key not in row:
                        raise SystemExit(f"{path}:{line_no}: missing {key}")
                row["_source"] = str(path)
                rows.append(row)
    seen = Counter((r["task_id"], r["seed"], r.get("corpus"), r["attempt"]) for r in rows)
    dupes = [k for k, v in seen.items() if v > 1]
    if dupes:
        print(f"warning: {len(dupes)} duplicate (task, seed, corpus, attempt) keys; "
              f"files from different runs share seeds, e.g. {dupes[0]}", file=sys.stderr)
    return rows


def is_error(row: dict) -> bool:
    return row.get("reward") is None


def group_key(row: dict, by_corpus: bool) -> tuple:
    return (row["task_id"], row["seed"], row.get("corpus")) if by_corpus else (row["task_id"], row["seed"])


def all_equal(rewards: list[float]) -> bool:
    return len({round(float(r), 9) for r in rewards}) <= 1


def summarize(rows: list[dict], by_corpus: bool = True) -> dict:
    """Stats for one cell. Groups are (task_id, seed[, corpus]) across attempts; a group needs
    two valid episodes to say anything about informativeness, so shorter ones are counted apart."""
    valid = [r for r in rows if not is_error(r)]
    groups = defaultdict(list)
    for r in valid:
        groups[group_key(r, by_corpus)].append(float(r["reward"]))
    usable = [g for g in groups.values() if len(g) >= 2]
    solves = sum(1 for r in valid if float(r["reward"]) == 1.0)
    informative = sum(1 for g in usable if not all_equal(g))
    steps = sum(int(r.get("steps", 0)) for r in valid)
    solve_post = posterior(solves, len(valid))
    return {
        "episodes": len(rows), "errors": len(rows) - len(valid), "valid": len(valid),
        "mean_reward": (sum(float(r["reward"]) for r in valid) / len(valid)) if valid else None,
        "solve": solve_post,
        "groups": len(usable), "short_groups": len(groups) - len(usable),
        "group_sizes": dict(sorted(Counter(len(g) for g in usable).items())),
        "informative": posterior(informative, len(usable)),
        "projected_informative_at_g": {"g": ARENA_GROUP, "value": projected_informative(
            solve_post["a"], solve_post["b"], ARENA_GROUP)},
        "invalid_per_step": (sum(int(r.get("invalid", 0)) for r in valid) / steps) if steps else None,
        "ends": dict(sorted(Counter(r.get("end", "") for r in rows).items())),
    }


def compute_stats(rows: list[dict]) -> dict:
    cells = defaultdict(list)
    per_corpus = defaultdict(list)
    for r in rows:
        cells[r["task_id"]].append(r)
        if r.get("corpus") is not None:
            per_corpus[(r["task_id"], r["corpus"])].append(r)
    return {"cells": {t: summarize(rs) for t, rs in sorted(cells.items())},
            "cell_corpus": {f"{t}|{c}": summarize(rs) for (t, c), rs in sorted(per_corpus.items())}}


def _fmt_post(p: dict) -> str:
    if p["n"] == 0:
        return "0/0 (no data)"
    return f"{p['k']}/{p['n']} {p['mean']:.2f} [{p['lo']:.2f},{p['hi']:.2f}]"


def print_stats(stats: dict, out=None) -> None:
    out = out or sys.stdout
    header = (f"{'cell':<42}{'eps':>4}{'err':>4} {'mean':>5}  {'solve k/n post [90%]':<26}"
              f"{'informative grp k/n post [90%]':<31}{'sizes':<7}{'p_inf@' + str(ARENA_GROUP):>8}"
              f"{'inv/step':>9}  ends")
    for title, table in (("per cell", stats["cells"]), ("per (cell, corpus function)", stats["cell_corpus"])):
        if not table:
            continue
        print(f"\n{title}  (Jeffreys Beta(0.5,0.5) posteriors; error rows excluded, not zeroed)", file=out)
        print(header, file=out)
        for name, s in table.items():
            mean = "-" if s["mean_reward"] is None else f"{s['mean_reward']:.2f}"
            inv = "-" if s["invalid_per_step"] is None else f"{s['invalid_per_step']:.3f}"
            proj = "-" if s["valid"] == 0 else f"{s['projected_informative_at_g']['value']:.2f}"
            sizes = ",".join(f"{k}x{v}" for k, v in s["group_sizes"].items()) or "-"
            ends = ",".join(f"{k}:{v}" for k, v in s["ends"].items())
            print(f"{name:<42}{s['episodes']:>4}{s['errors']:>4} {mean:>5}  {_fmt_post(s['solve']):<26}"
                  f"{_fmt_post(s['informative']):<31}{sizes:<7}{proj:>8}"
                  f"{inv:>9}  {ends}", file=out)


# ---------------------------------------------------------------- select

def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_cells(stats: dict, template: list[dict], max_cells: int, seed: int, saturated: float = 0.9,
                 dead: float = 0.05, min_groups: int = 1, objective: str = "group") -> tuple[list[dict], list[dict]]:
    """Thompson-sample the informative-group posterior over eligible cells. Returns the task list
    (template entries copied verbatim, in template order) and one reason record per cell."""
    if max_cells < 1 or max_cells > 50:
        raise ValueError("max_cells must be 1..50 (the arena limit)")
    by_id = {}
    for entry in template:
        if entry["task_id"] in by_id:
            raise ValueError(f"template lists {entry['task_id']} twice")
        by_id[entry["task_id"]] = entry
    cells = stats["cells"]
    reasons, eligible = [], []
    for task_id in sorted(set(cells) | set(by_id)):
        s = cells.get(task_id)
        record = {"task_id": task_id}
        if s is not None:
            record.update(solve=s["solve"], informative=s["informative"], errors=s["errors"],
                          projected_informative_at_g=s["projected_informative_at_g"])
        if s is None:
            record.update(decision="excluded", reason="no calibration data")
        elif task_id not in by_id:
            record.update(decision="excluded", reason="no template entry, so its limits are unknown")
        elif s["groups"] < min_groups:
            record.update(decision="excluded", reason=f"{s['groups']} usable groups < {min_groups}")
        elif s["solve"]["mean"] >= saturated:
            record.update(decision="excluded", reason=f"saturated: posterior mean solve {s['solve']['mean']:.3f} >= {saturated}")
        elif s["solve"]["mean"] <= dead:
            record.update(decision="excluded", reason=f"dead: posterior mean solve {s['solve']['mean']:.3f} <= {dead}")
        else:
            eligible.append(record)
        reasons.append(record)
    rng = np.random.default_rng(seed)
    for record in eligible:  # sorted order, so one seed gives one draw sequence
        if objective == "group":
            post = record["informative"]
            record["draw"] = float(rng.beta(post["a"], post["b"]))
        else:  # projected: draw p from the solve posterior, score the chance a group of g is mixed
            p = float(rng.beta(record["solve"]["a"], record["solve"]["b"]))
            record["draw"] = 1.0 - p ** ARENA_GROUP - (1.0 - p) ** ARENA_GROUP
    ranked = sorted(eligible, key=lambda r: (-r["draw"], r["task_id"]))
    for rank, record in enumerate(ranked, 1):
        record["rank"] = rank
        chosen = rank <= max_cells
        record["decision"] = "selected" if chosen else "excluded"
        record["reason"] = (f"Thompson draw {record['draw']:.3f} ({objective}) ranked {rank} of {len(ranked)} eligible"
                            + ("" if chosen else f"; beyond --max-cells {max_cells}"))
    chosen_ids = {r["task_id"] for r in ranked[:max_cells]}
    tasks = [copy.deepcopy(e) for e in template if e["task_id"] in chosen_ids]
    return tasks, reasons


# ---------------------------------------------------------------- ledger

GENESIS = "0" * 64


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


def parse_result(result: dict) -> tuple[str, dict]:
    """Accept {run_id, domains: {name: {solved, total}}}. The arena's own field names are
    mapped here once they are known; until then the caller writes this shape."""
    run_id = result.get("run_id") or result.get("id")
    domains = result.get("domains")
    if not run_id or not isinstance(domains, dict) or not domains:
        raise SystemExit("result must carry run_id and a non-empty domains object")
    clean = {}
    for name, value in domains.items():
        if not isinstance(value, dict) or not {"solved", "total"} <= set(value):
            raise SystemExit(f"domain {name!r} needs solved and total")
        solved, total = value["solved"], value["total"]
        if type(solved) is not int or type(total) is not int or not 0 <= solved <= total or total < 1:
            raise SystemExit(f"domain {name!r}: need integers 0 <= solved <= total, total >= 1")
        clean[name] = {"solved": solved, "total": total}
    return str(run_id), dict(sorted(clean.items()))


def ledger_add(ledger: Path, receipt_path: Path, result_path: Path, request_path: Path | None = None,
               now: str | None = None) -> dict:
    receipt = json.loads(receipt_path.read_text())
    if request_path is None:
        request_path = receipt_path.with_name(receipt_path.name.replace("receipt", "request", 1))
    request = json.loads(request_path.read_text())
    sha = canonical_sha(request)
    if sha != receipt.get("request_sha256"):
        raise SystemExit(f"{request_path} hashes to {sha}, not the receipt's {receipt.get('request_sha256')}")
    if request.get("submission_id") != receipt.get("submission_id"):
        raise SystemExit("request and receipt name different submission ids")
    run_id, domains = parse_result(json.loads(result_path.read_text()))
    entries = read_ledger(ledger)
    for e in entries:
        if e["submission_id"] == receipt["submission_id"] or e["run_id"] == run_id:
            raise SystemExit(f"ledger already holds {e['submission_id']} / {e['run_id']}; entries are never replaced")
    prev = hashlib.sha256(ledger.read_text().splitlines()[-1].encode()).hexdigest() if entries else GENESIS
    entry = {"seq": len(entries) + 1, "prev_sha256": prev,
             "recorded_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "submission_id": receipt["submission_id"], "request_sha256": sha, "run_id": run_id,
             "cells": [t["task_id"] for t in request.get("tasks", [])], "domains": domains,
             "solved": sum(d["solved"] for d in domains.values()),
             "total": sum(d["total"] for d in domains.values())}
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return entry


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


def ledger_show(entries: list[dict], baseline: dict[str, int], out=None) -> dict:
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
    return {"composite": composite, "best": best, "null_expected": expected,
            "null_p_value": tail(dist, composite), "runs": k}


# ---------------------------------------------------------------- onecell-search

STEP = {"max_steps": (4, 4, 32), "max_test_runs": (1, 0, 8), "completion_tokens": (2048, 1024, 16384)}
MIX_DELTA = 0.1


def knob_names(config: dict) -> list[str]:
    names = ["max_steps", "max_test_runs", "difficulty_mix"]
    return names + [f"completion_tokens:{t}" for t in sorted(config["completion_tokens"])]


def knob_value(config: dict, name: str):
    if name.startswith("completion_tokens:"):
        return config["completion_tokens"].get(name.split(":", 1)[1])
    return config.get(name)


def changed_knobs(a: dict, b: dict) -> list[str]:
    names = sorted(set(knob_names(a)) | set(knob_names(b)))
    return [n for n in names if knob_value(a, n) != knob_value(b, n)]


def validate_config(config: dict) -> None:
    for key in ("max_steps", "max_test_runs"):
        if type(config.get(key)) is not int:
            raise SystemExit(f"config.{key} must be an integer")
    mix = config.get("difficulty_mix")
    if not isinstance(mix, dict) or set(mix) != set(LEVELS) or abs(sum(mix.values()) - 1.0) > 1e-6 \
            or min(mix.values()) < 0:
        raise SystemExit("config.difficulty_mix needs d1, d2, d3 weights >= 0 summing to 1")
    if not isinstance(config.get("completion_tokens"), dict) or not config["completion_tokens"]:
        raise SystemExit("config.completion_tokens must map task ids to integers")


def candidate_moves(config: dict) -> list[tuple[str, object]]:
    """Every legal single-knob move, in a fixed order."""
    moves = []
    for key in ("max_steps", "max_test_runs"):
        step, lo, hi = STEP[key]
        moves += [(key, config[key] + s) for s in (-step, step) if lo <= config[key] + s <= hi]
    mix = config["difficulty_mix"]
    for src in LEVELS:
        for dst in LEVELS:
            if src != dst and mix[src] >= MIX_DELTA - 1e-12:
                new = dict(mix)
                new[src] = round(new[src] - MIX_DELTA, 6)
                new[dst] = round(new[dst] + MIX_DELTA, 6)
                moves.append(("difficulty_mix", new))
    step, lo, hi = STEP["completion_tokens"]
    for task_id, value in sorted(config["completion_tokens"].items()):
        moves += [(f"completion_tokens:{task_id}", value + s) for s in (-step, step) if lo <= value + s <= hi]
    return moves


def apply_move(config: dict, knob: str, value) -> dict:
    new = copy.deepcopy(config)
    if knob.startswith("completion_tokens:"):
        new["completion_tokens"][knob.split(":", 1)[1]] = value
    else:
        new[knob] = value
    return new


def pick_seed_range(rng, count: int, taken: set[int]) -> int:
    for _ in range(10_000):
        base = int(rng.integers(100, 1000)) * 10_000
        if not any(s in taken for s in range(base, base + count)):
            return base
    raise SystemExit("could not find a fresh seed range")


def propose(config: dict, seed: int, seen_seeds: set[int], history: list[dict], tasks: list[str],
            discovery_seeds: int = 8, confirm_seeds: int = 8, attempts: int = 4, margin: float = 0.05,
            confirm_tolerance: float = 0.0, metric: str = "mean_reward") -> dict:
    validate_config(config)
    tried = {(h["knob"], json.dumps(h["to"], sort_keys=True)) for h in history}
    moves = [m for m in candidate_moves(config) if (m[0], json.dumps(m[1], sort_keys=True)) not in tried]
    if not moves:
        raise SystemExit("every single-knob move from this config has been tried")
    rng = np.random.default_rng(seed)
    knob, value = moves[int(rng.integers(len(moves)))]
    candidate = apply_move(config, knob, value)
    assert changed_knobs(config, candidate) == [knob]
    taken = set(seen_seeds)
    disc = pick_seed_range(rng, discovery_seeds, taken)
    taken |= set(range(disc, disc + discovery_seeds))
    conf = pick_seed_range(rng, confirm_seeds, taken)
    split = lambda base, n: {"seed_base": base, "seeds": list(range(base, base + n)), "attempts": attempts,
                             "task_ids": tasks}  # noqa: E731
    return {
        "kind": "onecell-plan", "seed": seed, "knob": knob, "from": knob_value(config, knob), "to": value,
        "baseline": config, "candidate": candidate, "metric": metric,
        "rule": {"discovery": f"candidate - baseline > {margin} (strict gain)",
                 "confirmation": f"candidate - baseline >= {-confirm_tolerance} (no loss)",
                 "margin": margin, "confirm_tolerance": confirm_tolerance},
        "discovery": split(disc, discovery_seeds), "confirmation": split(conf, confirm_seeds),
        "pairing": "baseline and candidate run the SAME seeds and attempts within each split; "
                   "confirmation seeds are fresh (unseen in the supplied runs and disjoint from discovery)",
        "note": ("difficulty_mix only reweights levels, so baseline and candidate may be the same runs"
                 if knob == "difficulty_mix" else "the caller applies the knob before the candidate runs"),
    }


def level_of(task_id: str) -> str:
    match = re.search(r"-(d[123])$", task_id)
    if not match:
        raise SystemExit(f"cannot read a difficulty from {task_id!r}")
    return match.group(1)


def paired_rows(base: list[dict], cand: list[dict], split: dict) -> tuple[list, list, int]:
    seeds, task_ids = set(split["seeds"]), set(split["task_ids"])
    for rows, label in ((base, "baseline"), (cand, "candidate")):
        stray = [r for r in rows if r["seed"] not in seeds or r["task_id"] not in task_ids]
        if stray:
            raise SystemExit(f"{label} has {len(stray)} rows outside the plan's seeds or tasks; refusing to mix splits")
    key = lambda r: (r["task_id"], r["seed"], r.get("corpus"), r["attempt"])  # noqa: E731
    b, c = {key(r): r for r in base}, {key(r): r for r in cand}
    if set(b) != set(c):
        raise SystemExit(f"unpaired episodes: {len(set(b) ^ set(c))} keys appear on one side only")
    keep = [k for k in sorted(b, key=str) if not is_error(b[k]) and not is_error(c[k])]
    return [b[k] for k in keep], [c[k] for k in keep], len(b) - len(keep)


def split_metric(rows: list[dict], mix: dict, metric: str) -> float:
    """Mix-weighted metric over difficulty levels; levels with weight 0 do not count."""
    by_level = defaultdict(list)
    for r in rows:
        by_level[level_of(r["task_id"])].append(r)
    total = 0.0
    for level, weight in mix.items():
        if weight == 0:
            continue
        level_rows = by_level.get(level)
        if not level_rows:
            raise SystemExit(f"mix weights {level} at {weight} but the split has no {level} episodes")
        if metric == "mean_reward":
            value = sum(float(r["reward"]) for r in level_rows) / len(level_rows)
        elif metric == "solve_rate":
            value = sum(float(r["reward"]) == 1.0 for r in level_rows) / len(level_rows)
        else:  # informative_rate: groups of (task, seed, corpus) across attempts
            groups = defaultdict(list)
            for r in level_rows:
                groups[group_key(r, True)].append(float(r["reward"]))
            usable = [g for g in groups.values() if len(g) >= 2]
            value = sum(not all_equal(g) for g in usable) / len(usable) if usable else 0.0
        total += weight * value
    return total


def evaluate_split(base, cand, split, plan, rng, boot=2000) -> dict:
    b, c, dropped = paired_rows(base, cand, split)
    if not b:
        raise SystemExit("no valid paired episodes in a split")
    metric = plan["metric"]
    delta = split_metric(c, plan["candidate"]["difficulty_mix"], metric) - \
        split_metric(b, plan["baseline"]["difficulty_mix"], metric)
    clusters = sorted({(r["task_id"], r["seed"]) for r in b})  # resample whole (task, seed) clusters
    idx = defaultdict(list)
    for i, r in enumerate(b):
        idx[(r["task_id"], r["seed"])].append(i)
    draws = []
    for _ in range(boot):
        pick = [i for j in rng.integers(len(clusters), size=len(clusters)) for i in idx[clusters[j]]]
        try:
            draws.append(split_metric([c[i] for i in pick], plan["candidate"]["difficulty_mix"], metric)
                         - split_metric([b[i] for i in pick], plan["baseline"]["difficulty_mix"], metric))
        except SystemExit:  # a resample that misses a weighted level is skipped
            continue
    lo, hi = (np.quantile(draws, [0.05, 0.95]).tolist() if draws else [None, None])
    return {"pairs": len(b), "dropped_error_pairs": dropped, "delta": delta, "bootstrap_90": [lo, hi]}


def decide(plan: dict, base_disc, cand_disc, base_conf, cand_conf) -> dict:
    if plan.get("kind") != "onecell-plan":
        raise SystemExit("not an onecell plan")
    if changed_knobs(plan["baseline"], plan["candidate"]) != [plan["knob"]]:
        raise SystemExit("plan changes more than one knob")
    rng = np.random.default_rng(plan["seed"])
    disc = evaluate_split(base_disc, cand_disc, plan["discovery"], plan, rng)
    conf = evaluate_split(base_conf, cand_conf, plan["confirmation"], plan, rng)
    rule = plan["rule"]
    gain = disc["delta"] > rule["margin"]
    no_loss = conf["delta"] >= -rule["confirm_tolerance"]
    reasons = []
    if not gain:
        reasons.append(f"discovery delta {disc['delta']:+.4f} is not > margin {rule['margin']}")
    if not no_loss:
        reasons.append(f"confirmation delta {conf['delta']:+.4f} is a loss beyond {rule['confirm_tolerance']}")
    return {"knob": plan["knob"], "from": plan["from"], "to": plan["to"], "metric": plan["metric"],
            "accept": gain and no_loss, "discovery": disc, "confirmation": conf,
            "reasons": reasons or ["strict discovery gain and no confirmation loss"]}


def init_config(template: list[dict]) -> dict:
    tree = ast.parse((ROOT / "qe_arena" / "environment.py").read_text())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    levels = Counter(level_of(e["task_id"]) for e in template)
    mix = {lv: round(levels.get(lv, 0) / len(template), 6) for lv in LEVELS}
    mix[max(mix, key=mix.get)] += round(1.0 - sum(mix.values()), 6)
    return {"max_steps": consts["MAX_STEPS"], "max_test_runs": consts["MAX_TEST_RUNS"], "difficulty_mix": mix,
            "completion_tokens": {e["task_id"]: e["completion_tokens"] for e in template}}


# ---------------------------------------------------------------- CLI

def parse_baseline(text: str) -> dict[str, int]:
    out = {}
    for part in filter(None, text.split(",")):
        name, _, value = part.partition("=")
        out[name.strip()] = int(value)
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def main(argv=None) -> int:
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
    p.add_argument("--saturated", type=float, default=0.9, help="exclude posterior mean solve >= this")
    p.add_argument("--dead", type=float, default=0.05, help="exclude posterior mean solve <= this")
    p.add_argument("--min-groups", type=int, default=1)
    p.add_argument("--objective", choices=["group", "projected"], default="group",
                   help=f"group: informative-group posterior; projected: P(mixed) for {ARENA_GROUP} independent episodes")
    p = sub.add_parser("ledger", help="append-only submission ledger")
    lsub = p.add_subparsers(dest="action", required=True)
    for name in ("add", "show"):
        q = lsub.add_parser(name)
        q.add_argument("--ledger", type=Path, default=ROOT / "submission" / "ledger.jsonl")
        if name == "add":
            q.add_argument("--receipt", type=Path, required=True)
            q.add_argument("--result", type=Path, required=True)
            q.add_argument("--request", type=Path, help="default: the receipt's sibling request-vN.json")
        else:
            q.add_argument("--baseline", default="industrial=2,office=2", help="per-domain solved count of the baseline")
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
    q.add_argument("--discovery-seeds", type=int, default=8)
    q.add_argument("--confirm-seeds", type=int, default=8)
    q.add_argument("--attempts", type=int, default=4)
    q.add_argument("--margin", type=float, default=0.05)
    q.add_argument("--confirm-tolerance", type=float, default=0.0)
    q.add_argument("--metric", choices=["mean_reward", "solve_rate", "informative_rate"], default="mean_reward")
    q.add_argument("--out", type=Path, required=True)
    q = osub.add_parser("decide")
    q.add_argument("--plan", type=Path, required=True)
    for name in ("baseline-discovery", "candidate-discovery", "baseline-confirm", "candidate-confirm"):
        q.add_argument(f"--{name}", type=Path, required=True)
    q.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "stats":
        stats = compute_stats(load_runs(args.runs))
        print_stats(stats)
        if args.json:
            write_json(args.json, stats)
    elif args.cmd == "select":
        stats = compute_stats(load_runs(args.runs))
        template = json.loads(args.template.read_text())
        tasks, reasons = select_cells(stats, template, args.max_cells, args.seed, args.saturated, args.dead,
                                      args.min_groups, args.objective)
        write_json(args.out, tasks)
        why = args.why or args.out.with_suffix(".why.json")
        write_json(why, {"seed": args.seed, "max_cells": args.max_cells, "objective": args.objective,
                         "saturated": args.saturated, "dead": args.dead, "min_groups": args.min_groups,
                         "template": {"path": str(args.template), "sha256": file_sha(args.template)},
                         "runs": [{"path": str(r), "sha256": file_sha(r)} for r in args.runs], "cells": reasons})
        for r in reasons:
            print(f"{r['decision']:<9} {r['task_id']:<22} {r['reason']}")
        print(f"\n{len(tasks)} cells -> {args.out}; reasons -> {why}")
    elif args.cmd == "ledger" and args.action == "add":
        entry = ledger_add(args.ledger, args.receipt, args.result, args.request)
        print(json.dumps(entry, indent=2))
    elif args.cmd == "ledger":
        summary = ledger_show(read_ledger(args.ledger), parse_baseline(args.baseline))
        if args.json:
            write_json(args.json, summary)
    elif args.action == "init":
        write_json(args.out, init_config(json.loads(args.template.read_text())))
        print(args.out.read_text())
    elif args.action == "propose":
        config = json.loads(args.config.read_text())
        seen = {r["seed"] for r in load_runs(args.runs)} if args.runs else set()
        history = [json.loads(l) for l in args.history.read_text().splitlines() if l.strip()] \
            if args.history and args.history.exists() else []
        tasks = sorted(config["completion_tokens"])
        plan = propose(config, args.seed, seen, history, tasks, args.discovery_seeds, args.confirm_seeds,
                       args.attempts, args.margin, args.confirm_tolerance, args.metric)
        write_json(args.out, plan)
        print(f"knob {plan['knob']}: {plan['from']} -> {plan['to']}; discovery seeds "
              f"{plan['discovery']['seed_base']}+{len(plan['discovery']['seeds'])}, confirmation "
              f"{plan['confirmation']['seed_base']}+{len(plan['confirmation']['seeds'])} -> {args.out}")
    else:
        plan = json.loads(args.plan.read_text())
        result = decide(plan, *(load_runs([path]) for path in (
            args.baseline_discovery, args.candidate_discovery, args.baseline_confirm, args.candidate_confirm)))
        print(json.dumps(result, indent=2))
        if args.out:
            write_json(args.out, result)
        return 0 if result["accept"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
