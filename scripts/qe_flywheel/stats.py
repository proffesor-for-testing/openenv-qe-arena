"""Calibration statistics: Jeffreys posteriors, GRPO groups, and error rows kept apart."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict

PRIOR = 0.5  # Jeffreys: Beta(0.5, 0.5)
CI = 0.90
# Arena AGENTS.md: each optimizer step plays 5 episodes of one task and trains on the first 4 that end
# with a valid reward. The contract carries no seed, and our reset draws a fresh instance without one,
# so the trainer's group is G independent instances, not G attempts at one seed. Both sizes are shown.
TRAIN_GROUP = 4
PROJECTED_GROUPS = (4, 5)


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
        "projected_informative": {str(g): projected_informative(solve_post["a"], solve_post["b"], g)
                                  for g in PROJECTED_GROUPS},
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
              f"{'same-seed inf k/n post [90%]':<31}{'sizes':<7}"
              + "".join(f"{'p_inf@' + str(g):>8}" for g in PROJECTED_GROUPS) + f"{'inv/step':>9}  ends")
    legend = ("same-seed inf: measured share of (task, seed[, corpus]) groups whose rewards differ; "
              "p_inf@G: projected chance that G independent instances are mixed (solved vs not), "
              "from the solve posterior. The arena trains on 4 of 5 unseeded episodes.")
    for title, table in (("per cell", stats["cells"]), ("per (cell, corpus function)", stats["cell_corpus"])):
        if not table:
            continue
        print(f"\n{title}  (Jeffreys Beta(0.5,0.5) posteriors; error rows excluded, not zeroed)", file=out)
        print(legend, file=out)
        print(header, file=out)
        for name, s in table.items():
            mean = "-" if s["mean_reward"] is None else f"{s['mean_reward']:.2f}"
            inv = "-" if s["invalid_per_step"] is None else f"{s['invalid_per_step']:.3f}"
            proj = "".join(f"{'-' if s['valid'] == 0 else format(v, '.2f'):>8}"
                           for v in s["projected_informative"].values())
            sizes = ",".join(f"{k}x{v}" for k, v in s["group_sizes"].items()) or "-"
            ends = ",".join(f"{k}:{v}" for k, v in s["ends"].items())
            print(f"{name:<42}{s['episodes']:>4}{s['errors']:>4} {mean:>5}  {_fmt_post(s['solve']):<26}"
                  f"{_fmt_post(s['informative']):<31}{sizes:<7}{proj}"
                  f"{inv:>9}  {ends}", file=out)
