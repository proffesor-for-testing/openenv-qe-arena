"""`ledger show`: per-run scores and the best-per-domain composite, each against its nulls.

Primary: the empirical null, other entrants' runs on the same benchmark. Sensitivity: the model-based
null of an untrained baseline (null.py), which is generous because its Jeffreys prior puts about 8% on
the domains the baseline scored 0 in, while trained runs mostly score below the baseline."""

from __future__ import annotations

import sys

from . import empirical, null
from .ledger import DOMAIN_IDS, null_domains, parse_result

SHORT = {"software-engineering": "soft", "industrial-physical-systems": "ind", "natural-science": "sci",
         "office-white-collar": "off", "finance-economics": "fin", "mathematics-or-formal-reasoning": "math",
         "media-content-production": "media", "cybersecurity": "cyber"}


def ledger_show(entries: list[dict], baseline: dict, population: dict | None = None, out=None, source: str = "",
                include_manual: bool = False, head: dict | None = None) -> dict:
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
    pop = None
    if population is not None:
        users = {e["record"]["user"] for e in entries}
        pop = empirical.population(population["runs"], benchmarks[0], users, parse_result)
        s = empirical.summary(pop)
        print(f"EMPIRICAL NULL: {s['runs']} other runs on {benchmarks[0]} (fetched {population.get('fetched_at')}; "
              f"excluding {sorted(users)}{'; ' + str(pop['skipped']) + ' malformed skipped' if pop['skipped'] else ''}): "
              f"mean {s['mean']:.2f}/40, sd {s['sd']:.2f}, range {s['min']}-{s['max']}", file=out)
        print("  per-domain task rate: " + ", ".join(f"{SHORT[n]} {s['rates'][n]:.3f}" for n in DOMAIN_IDS), file=out)
    else:
        print("EMPIRICAL NULL unavailable: no leaderboard population saved (run `ledger fetch`)", file=out)
    print(f"\nPER-RUN SCORES ({benchmarks[0]}): each row is one noisy single attempt on the held-out set", file=out)
    print(f"{'seq':>3}  {'submission':<28}{'run':<26}{'total':>7}  {'pctile':>6}  {'P(other>=)':>10}  "
          f"{'model P(>=)':>11}", file=out)
    per_run = {}
    for e in entries:
        row = {"model_p": null.tail(single, e["solved"])}
        if pop is not None:
            row.update(empirical.run_position(pop, e["solved"]), deltas=empirical.domain_deltas(pop, e["domains"]))
        per_run[e["run_id"]] = row
        emp = f"{row['percentile']:>6.1f}  {row['p_other_ge']:>10.3f}" if pop is not None else f"{'-':>6}  {'-':>10}"
        print(f"{e['seq']:>3}  {e['submission_id']:<28}{e['run_id']:<26}{e['solved']:>3}/{e['total']:<3}  {emp}  "
              f"{row['model_p']:>11.3f}", file=out)
        if pop is not None:
            print("      vs an average other run: " + ", ".join(f"{SHORT[n]} {d:+.2f}" for n, d in row["deltas"].items()),
                  file=out)
    best = {n: max(e["domains"][n]["solved"] for e in entries) for n in DOMAIN_IDS}
    composite, k = sum(best.values()), len(entries)
    print(f"\nBEST-PER-DOMAIN COMPOSITE (not a run score; selection-inflated): {composite}  "
          + " + ".join(f"{n} {v}" for n, v in best.items() if v), file=out)
    emp_null = None
    if pop is not None:
        emp_null = empirical.composite_null(pop, k, composite)
        print(f"empirical null, best-per-domain of {k} other run(s) drawn with replacement ({emp_null['method']}): "
              f"E = {emp_null['expected']:.2f}, P(>= {composite}) = {emp_null['p_ge']:.3f}", file=out)
    model, plugin = null.composite_pmf(k, domains), null.composite_pmf(k, domains, "plugin")
    sens = {z: null.composite_pmf(k, domains, p_zero=z) for z in (0.0, 0.02, 0.05)}
    print(f"sensitivity, model-based null of an UNTRAINED baseline ({sum(d[1] for d in domains.values())}/"
          f"{sum(d[0] for d in domains.values())}, Jeffreys beta-binomial per domain; generous): "
          f"E = {null.mean(model):.2f}, P(>= {composite}) = {null.tail(model, composite):.3f}", file=out)
    print("  with the baseline's 0/5 domains fixed at p_zero: "
          + "; ".join(f"p_zero={z}: E {null.mean(d):.2f}, P(>= {composite}) {null.tail(d, composite):.3f}"
                      for z, d in sens.items()), file=out)
    print(f"  plug-in binomial at s/n (lower reference): E = {null.mean(plugin):.2f}, "
          f"P(>= {composite}) = {null.tail(plugin, composite):.3f}", file=out)
    if source:
        print(f"baseline source: {source}", file=out)
    return {"composite": composite, "best": best, "runs": k, "empirical": emp_null, "per_run": per_run,
            "model_expected": null.mean(model), "model_p_value": null.tail(model, composite),
            "plugin_expected": null.mean(plugin), "sensitivity": {str(z): null.tail(d, composite) for z, d in sens.items()}}
