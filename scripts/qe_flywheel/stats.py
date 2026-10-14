"""Calibration statistics: Jeffreys posteriors, GRPO groups, and error rows kept apart.

Rows with ``reward: null`` (harness errors) and ``global_budget`` rows (the whole calibration run ran
out of tokens) are artefacts of our harness: counted, reported apart and excluded, never scored as 0.
``context_budget`` and ``completion_budget`` depend on the limits the episode ran under: at or above the
submitted template's limits the arena would hit them too, so they are real failures (reward 0); under
those limits they are artefacts and are excluded with a warning. ``episode_status`` is the one rule,
shared by stats, select and search."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict

from qe_arena.domains import BY_SLUG, SOFTWARE

PRIOR = 0.5  # Jeffreys: Beta(0.5, 0.5)
CI = 0.90
# Arena AGENTS.md: each optimizer step plays 5 episodes of one task and trains on the first 4 that end
# with a valid reward. The contract carries no seed, and our reset draws a fresh instance without one,
# so the trainer's group is G independent instances, not G attempts at one seed. Both sizes are shown.
TRAIN_GROUP = 4
PROJECTED_GROUPS = (4, 5)
BUDGET_ENDS = frozenset({"global_budget", "context_budget", "completion_budget"})
# calibrate.py's documented per-episode defaults, used when a row does not record the limits it ran under
CALIBRATE_LIMITS = {"completion_tokens": 4096, "context_tokens": 8192}
ROW_LIMIT_FIELDS = {"completion_tokens": "episode_completion_tokens", "context_tokens": "episode_context_tokens"}
BUDGET_LIMIT = {"completion_budget": "completion_tokens", "context_budget": "context_tokens"}


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



def expected_all_equal(alphas, g: int) -> float:
    """E[sum_v q_v^g] for q ~ Dirichlet(alphas): the chance that g independent episodes all land on
    the same reward value. Each marginal q_v is Beta(alpha_v, alpha_0 - alpha_v)."""
    total = float(sum(alphas))
    return sum(math.prod((a + i) / (total + i) for i in range(g)) for a in alphas)


def reward_categories(rewards) -> tuple[list[float], list[float]]:
    """Distinct reward values (always including 0 and 1) and their Jeffreys-style Dirichlet alphas.
    With only 0/1 rewards this is exactly the Beta(k + .5, n - k + .5) solve posterior."""
    counts = Counter(round(float(r), 9) for r in rewards)
    values = sorted(set(counts) | {0.0, 1.0})
    return values, [counts.get(v, 0) + PRIOR for v in values]


def projected_mixed(rewards, g: int) -> float:
    """Posterior mean chance that g fresh instances do NOT all get the same reward, partial credit
    included: a mutant-kill group of [0.5, 0, 0, 0] has an advantage signal just as [1, 0, 0, 0] does."""
    return 1.0 - expected_all_equal(reward_categories(rewards)[1], g)


def load_runs(paths) -> list[dict]:
    """Read calibrate.py JSONL. A repeated (task, seed, corpus, attempt) key is a hard error: two
    files from different runs that share seeds would otherwise merge into one fake group."""
    rows, origin = [], {}
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
                key = (row["task_id"], row["seed"], row.get("corpus"), row["attempt"])
                if key in origin:
                    raise SystemExit(f"{path}:{line_no}: duplicate (task, seed, corpus, attempt) {key}, "
                                     f"first seen at {origin[key]}; runs that share seeds cannot be pooled")
                origin[key] = f"{path}:{line_no}"
                row["_source"] = str(path)
                rows.append(row)
    return rows


def is_error(row: dict) -> bool:
    return row.get("reward") is None


def episode_status(row: dict, limits: dict | None) -> str:
    """'error', 'harness_budget' (global_budget), 'under_limit' (a per-episode budget end hit under the
    submitted ``limits``, or with no limits to compare), 'budget_failure' (hit at or above them: the
    arena would fail too) or 'valid'."""
    if is_error(row):
        return "error"
    end = row.get("end")
    if end == "global_budget":
        return "harness_budget"
    if end in BUDGET_LIMIT:
        key = BUDGET_LIMIT[end]
        ran = row.get(ROW_LIMIT_FIELDS[key], CALIBRATE_LIMITS[key])
        submitted = (limits or {}).get(key)
        return "under_limit" if submitted is None or ran < submitted else "budget_failure"
    return "valid"


def is_excluded(row: dict, limits: dict | None = None) -> bool:
    return episode_status(row, limits) in ("error", "harness_budget", "under_limit")


CELL_RE = re.compile(r"(qe_[a-z_]+)-(?:([a-z]+)-)?(d\d)")


def cell_domain(task_id: str) -> str:
    """The arena domain a cell draws its corpus from: the slug between family and level, none for software."""
    match = CELL_RE.fullmatch(task_id)
    if match is None:
        return "unknown"
    return SOFTWARE if match.group(2) is None else BY_SLUG.get(match.group(2), "unknown")


def base_cell(task_id: str) -> str:
    """The software-engineering cell with the same family and level (the task ID without its domain slug)."""
    match = CELL_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"not a task cell: {task_id!r}")
    return f"{match.group(1)}-{match.group(3)}"


def template_limits(template: list[dict] | None) -> dict:
    return {e["task_id"]: {k: e.get(k) for k in CALIBRATE_LIMITS} for e in (template or [])}


def group_key(row: dict, by_corpus: bool) -> tuple:
    return (row["task_id"], row["seed"], row.get("corpus")) if by_corpus else (row["task_id"], row["seed"])


def all_equal(rewards: list[float]) -> bool:
    return len({round(float(r), 9) for r in rewards}) <= 1


def summarize(rows: list[dict], by_corpus: bool = True, limits: dict | None = None,
              limits_by_cell: dict | None = None) -> dict:
    """Stats for one cell, or for several when ``limits_by_cell`` gives each row its own cell's limits.
    Same-seed groups are (task_id, seed[, corpus]) across attempts and need two valid episodes; the
    projection treats every valid episode as one fresh instance."""
    def row_limits(r: dict) -> dict | None:
        return limits if limits_by_cell is None else limits_by_cell.get(r["task_id"])

    status = Counter(episode_status(r, row_limits(r)) for r in rows)
    valid = [r for r in rows if not is_excluded(r, row_limits(r))]
    rewards = [float(r["reward"]) for r in valid]
    groups = defaultdict(list)
    for r in valid:
        groups[group_key(r, by_corpus)].append(float(r["reward"]))
    usable = [g for g in groups.values() if len(g) >= 2]
    solves = sum(1 for x in rewards if x == 1.0)
    steps = sum(int(r.get("steps", 0)) for r in valid)
    return {
        "episodes": len(rows), "errors": status["error"], "budget": status["harness_budget"] + status["under_limit"],
        "under_limit": status["under_limit"], "budget_failures": status["budget_failure"],
        "valid": len(valid), "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
        "partial": sum(1 for x in rewards if 0.0 < x < 1.0),
        "solve": posterior(solves, len(valid)),
        "groups": len(usable), "short_groups": len(groups) - len(usable),
        "group_sizes": dict(sorted(Counter(len(g) for g in usable).items())),
        "informative": posterior(sum(1 for g in usable if not all_equal(g)), len(usable)),
        "projected_mixed": {str(g): projected_mixed(rewards, g) for g in PROJECTED_GROUPS},
        "reward_dirichlet": dict(zip(("values", "alphas"), reward_categories(rewards))),
        "invalid_per_step": (sum(int(r.get("invalid", 0)) for r in valid) / steps) if steps else None,
        "ends": dict(sorted(Counter(r.get("end", "") for r in rows).items())),
    }


def compute_stats(rows: list[dict], template: list[dict] | None = None) -> dict:
    """Per cell, per arena domain (all its cells pooled) and per (cell, corpus). ``template`` supplies the
    submitted limits that decide whether a per-episode budget end is a failure or an artefact; without it
    every such end is an artefact."""
    limits = template_limits(template)
    cells = defaultdict(list)
    domains = defaultdict(list)
    per_corpus = defaultdict(list)
    for r in rows:
        cells[r["task_id"]].append(r)
        domains[cell_domain(r["task_id"])].append(r)
        if r.get("corpus") is not None:
            per_corpus[(r["task_id"], r["corpus"])].append(r)
    out = {"cells": {t: summarize(rs, limits=limits.get(t)) for t, rs in sorted(cells.items())},
           "domains": {d: summarize(rs, limits_by_cell=limits) for d, rs in sorted(domains.items())},
           "cell_corpus": {f"{t}|{c}": summarize(rs, limits=limits.get(t)) for (t, c), rs in sorted(per_corpus.items())}}
    under = sum(c["under_limit"] for c in out["cells"].values())
    if under:
        print(f"warning: {under} context/completion budget end(s) ran under the submitted limits (or no template "
              "was given) and are excluded; calibrate at the template's limits to count them as failures",
              file=sys.stderr)
    return out


def _fmt_post(p: dict) -> str:
    if p["n"] == 0:
        return "0/0 (no data)"
    return f"{p['k']}/{p['n']} {p['mean']:.2f} [{p['lo']:.2f},{p['hi']:.2f}]"


LEGEND = ("err = reward null, bud = global budget or a per-episode budget hit under the submitted limits; both "
          "excluded, never 0 (a budget end at the submitted limits is a real failure). same-seed inf: measured "
          "share of (task, seed[, corpus]) groups whose rewards differ. p_mix@G: posterior mean chance that G "
          "fresh instances do not all get the same reward (partial credit counts). The arena trains on 4 of 5.")


def print_stats(stats: dict, out=None) -> None:
    out = out or sys.stdout
    header = (f"{'cell':<42}{'eps':>4}{'err':>4}{'bud':>4} {'mean':>5}  {'solve k/n post [90%]':<26}"
              f"{'same-seed inf k/n post [90%]':<31}{'sizes':<7}"
              + "".join(f"{'p_mix@' + str(g):>8}" for g in PROJECTED_GROUPS) + f"{'inv/step':>9}  ends")
    for title, table in (("per cell", stats["cells"]), ("per arena domain (its cells pooled)", stats.get("domains", {})),
                         ("per (cell, corpus function)", stats["cell_corpus"])):
        if not table:
            continue
        print(f"\n{title}  (Jeffreys posteriors)\n{LEGEND}", file=out)
        print(header, file=out)
        for name, s in table.items():
            mean = "-" if s["mean_reward"] is None else f"{s['mean_reward']:.2f}"
            inv = "-" if s["invalid_per_step"] is None else f"{s['invalid_per_step']:.3f}"
            proj = "".join(f"{'-' if s['valid'] == 0 else format(v, '.2f'):>8}"
                           for v in s["projected_mixed"].values())
            sizes = ",".join(f"{k}x{v}" for k, v in s["group_sizes"].items()) or "-"
            ends = ",".join(f"{k}:{v}" for k, v in s["ends"].items())
            print(f"{name:<42}{s['episodes']:>4}{s['errors']:>4}{s['budget']:>4} {mean:>5}  "
                  f"{_fmt_post(s['solve']):<26}{_fmt_post(s['informative']):<31}{sizes:<7}{proj}"
                  f"{inv:>9}  {ends}", file=out)
