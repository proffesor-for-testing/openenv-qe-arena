"""Darwin-style one-knob search: one change, paired discovery, fresh-seed confirmation."""

from __future__ import annotations

import ast
import copy
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .stats import all_equal, group_key, is_error

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("d1", "d2", "d3")


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
