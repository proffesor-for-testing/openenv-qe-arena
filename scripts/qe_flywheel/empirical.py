"""The empirical null: how other entrants' trained runs on the same benchmark actually score.

The model-based null (null.py) asks what an UNTRAINED model would score; the leaderboard says trained
runs usually score below the untrained baseline (on 2026-10-12, 38 other heldout-v2 runs averaged 2.95
of 40 against the baseline's 4), so a run is better compared with other runs. The population is every
leaderboard run of the same benchmark except the listed users' own."""

from __future__ import annotations

import itertools
import math

import numpy as np

EXACT_LIMIT = 200_000  # enumerate every k-tuple of other runs below this many; bootstrap above it
BOOT = 20_000


def population(runs: list[dict], benchmark: str, exclude_users, parse, exclude_runs=()) -> dict:
    """Per-domain solved counts of the other runs. ``parse`` validates a record (ledger.parse_result);
    records that fail it are counted, not used. Runs by ``exclude_users`` or with an id in
    ``exclude_runs`` are left out."""
    exclude, skip_ids = set(exclude_users), set(exclude_runs)
    kept, skipped = [], 0
    for run in runs:
        if not isinstance(run, dict):
            skipped += 1
            continue
        if run.get("benchmark") != benchmark or run.get("user") in exclude or run.get("run_id") in skip_ids:
            continue
        try:
            run_id, domains = parse(run)
        except SystemExit:
            skipped += 1
            continue
        kept.append({"run_id": run_id, "user": run["user"], "domains": domains})
    if not kept:
        raise SystemExit(f"no other runs on {benchmark} to compare with")
    names = list(kept[0]["domains"])
    totals = {n: kept[0]["domains"][n]["total"] for n in names}
    solved = np.array([[r["domains"][n]["solved"] for n in names] for r in kept])
    return {"benchmark": benchmark, "names": names, "totals": totals, "solved": solved, "skipped": skipped,
            "runs": [r["run_id"] for r in kept]}


def summary(pop: dict) -> dict:
    totals = pop["solved"].sum(axis=1)
    n_runs = len(totals)
    rates = {n: float(pop["solved"][:, i].sum()) / (pop["totals"][n] * n_runs) for i, n in enumerate(pop["names"])}
    return {"runs": n_runs, "mean": float(totals.mean()), "sd": float(totals.std(ddof=1)) if n_runs > 1 else 0.0,
            "min": int(totals.min()), "max": int(totals.max()), "rates": rates}


def run_position(pop: dict, solved_total: int) -> dict:
    """Mid-rank percentile among the other runs, and P(a random other run scores >= this)."""
    totals = pop["solved"].sum(axis=1)
    below, equal = float((totals < solved_total).mean()), float((totals == solved_total).mean())
    return {"percentile": 100.0 * (below + equal / 2.0), "p_other_ge": float((totals >= solved_total).mean())}


def zero_domain_prior(pop: dict) -> dict:
    """Empirical-Bayes Beta prior per domain: Jeffreys posterior of the other runs' pooled solves."""
    n_runs = len(pop["solved"])
    return {name: (float(pop["solved"][:, i].sum()) + 0.5,
                   pop["totals"][name] * n_runs - float(pop["solved"][:, i].sum()) + 0.5)
            for i, name in enumerate(pop["names"])}


def domain_deltas(pop: dict, domains: dict) -> dict:
    """This run's solved count per domain minus what an average other run solves there."""
    rates = summary(pop)["rates"]
    return {n: domains[n]["solved"] - pop["totals"][n] * rates[n] for n in pop["names"]}


def composite_null(pop: dict, k: int, observed: int, seed: int = 0) -> dict:
    """Best-per-domain composite of k DISTINCT other runs (drawn without replacement, as k real
    submissions are k different runs). Exact over every k-subset when there are few enough, otherwise
    a seeded Monte Carlo of random k-subsets."""
    solved, n = pop["solved"], len(pop["solved"])
    if k > n:
        raise SystemExit(f"cannot draw {k} distinct runs from {n} other runs")
    if math.comb(n, k) <= EXACT_LIMIT:
        idx = np.array(list(itertools.combinations(range(n), k)), dtype=int).reshape(-1, k)
        method = f"exact over all C({n},{k}) subsets"
    else:
        idx = np.argsort(np.random.default_rng(seed).random((BOOT, n)), axis=1)[:, :k]
        method = f"Monte Carlo, {BOOT} random subsets"
    values = solved[idx].max(axis=1).sum(axis=1)
    return {"expected": float(values.mean()), "p_ge": float((values >= observed).mean()), "method": method,
            "k": k, "observed": observed, "population": n}

