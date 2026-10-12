"""Darwin-style one-knob search: one change, paired discovery, fresh-seed confirmation.

The default metric is the trainer's own: the chance that a group of G fresh instances of a cell gets
rewards that are not all equal (GRPO's only source of gradient). A leniency knob (more steps, tokens or
test runs) or a mix shift toward easy cells raises solve rates, but on this metric it wins only where it
moves a cell toward mixed outcomes; pushing an easy cell further toward always-solved loses. Every
decision is gated on a paired bootstrap lower bound, not on a point estimate."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .stats import TRAIN_GROUP, episode_status

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("d1", "d2", "d3")
METRICS = ("mixed_rate", "mean_reward", "solve_rate")
BOOT = 2000
ALPHA = 0.05  # total one-sided error budget per move, spent across its tests (see alpha_for)
SEEDS = 64  # per split: the review measured >= 80% power for a 3-cell gain at 64, about 34% at 32
# Fields that differ between two genuine runs of the same episode; `seconds` and the like do not count.
STABLE = ("reward", "end", "steps", "invalid", "completion_tokens", "prompt_tokens_last", "corpus")


def alpha_for(retest: int) -> float:
    """Alpha spending for repeated tests of one move: ALPHA/2, ALPHA/4, ... sums to ALPHA."""
    return ALPHA * 0.5 ** (retest + 1)


def move_key(knob: str, value) -> tuple:
    return knob, json.dumps(value, sort_keys=True)


def history_status(history: list[dict]) -> tuple[set, Counter]:
    """Blocked moves and retest counts from past ``decide`` results. A move is blocked when it was
    accepted, or rejected with the discovery bootstrap UPPER bound below the margin (it cannot be a gain).
    Any other rejection was inconclusive: the move may be tested again, at a smaller alpha."""
    blocked, retests = set(), Counter()
    for h in history:
        key = move_key(h["knob"], h["to"])
        disc = h.get("discovery") or {}
        if h.get("accept") or ("upper_bound" in disc and disc["upper_bound"] < h.get("margin", 0.0)):
            blocked.add(key)
        else:
            retests[key] += 1
    return blocked, retests


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


def plan_sha(plan: dict) -> str:
    """Catches an accidental edit between propose and decide; anyone can recompute it after a deliberate one."""
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def propose(config: dict, seed: int, seen_seeds: set[int], history: list[dict], tasks: list[str],
            discovery_seeds: int = SEEDS, confirm_seeds: int = SEEDS, attempts: int = 1, margin: float = 0.0,
            confirm_tolerance: float = 0.05, metric: str = "mixed_rate", group_size: int = TRAIN_GROUP,
            error_tolerance: float = 0.02) -> dict:
    validate_config(config)
    if metric not in METRICS:
        raise SystemExit(f"metric must be one of {METRICS}")
    if metric == "mixed_rate" and attempts != 1:
        raise SystemExit("mixed_rate treats every episode as a fresh instance; use --attempts 1")
    if min(discovery_seeds, confirm_seeds) * attempts < group_size:
        raise SystemExit(f"each task needs at least {group_size} episodes per split")
    blocked, retests = history_status(history)
    moves = [m for m in candidate_moves(config) if move_key(*m) not in blocked]
    if metric != "mixed_rate":  # a mix shift toward easy cells wins on reward without any training
        moves = [m for m in moves if m[0] != "difficulty_mix"]
    if not moves:
        raise SystemExit("every single-knob move from this config has been accepted or ruled out")
    rng = np.random.default_rng(seed)
    knob, value = moves[int(rng.integers(len(moves)))]
    candidate = apply_move(config, knob, value)
    assert changed_knobs(config, candidate) == [knob]
    retest = retests[move_key(knob, value)]
    alpha = alpha_for(retest)
    taken = set(seen_seeds)
    disc = pick_seed_range(rng, discovery_seeds, taken)
    taken |= set(range(disc, disc + discovery_seeds))
    conf = pick_seed_range(rng, confirm_seeds, taken)
    split = lambda base, n: {"seed_base": base, "seeds": list(range(base, base + n)), "attempts": attempts,
                             "task_ids": list(tasks)}  # noqa: E731
    plan = {
        "kind": "onecell-plan", "seed": seed, "knob": knob, "from": knob_value(config, knob), "to": value,
        "baseline": config, "candidate": candidate, "metric": metric, "group_size": group_size,
        "retest": retest,
        "rule": {"discovery": f"paired-bootstrap {alpha:.2%} lower bound of candidate - baseline > {margin}",
                 "confirmation": f"paired-bootstrap {alpha:.2%} lower bound > {-confirm_tolerance}",
                 "margin": margin, "confirm_tolerance": confirm_tolerance, "alpha": alpha, "boot": BOOT,
                 "error_tolerance": error_tolerance},
        "discovery": split(disc, discovery_seeds), "confirmation": split(conf, confirm_seeds),
        "pairing": "baseline and candidate are independent runs on the SAME seeds and attempts within each "
                   "split; confirmation seeds are fresh (unseen in the supplied runs and disjoint from discovery)",
        "note": "the caller applies the knob before the candidate runs; for difficulty_mix both arms run the "
                "same cells as independent runs and the mix reweights them",
    }
    plan["plan_sha256"] = plan_sha(plan)
    return plan


def level_of(task_id: str) -> str:
    match = re.search(r"-(d[123])$", task_id)
    if not match:
        raise SystemExit(f"cannot read a difficulty from {task_id!r}")
    return match.group(1)


def index_arm(rows: list[dict], split: dict, label: str) -> dict:
    """Map (task, seed, attempt) -> row; refuse duplicates, strays and missing keys."""
    want = {(t, s, a) for t in split["task_ids"] for s in split["seeds"] for a in range(split["attempts"])}
    out = {}
    for r in rows:
        key = (r["task_id"], r["seed"], r["attempt"])
        if key not in want:
            raise SystemExit(f"{label} has row {key} outside the plan's tasks, seeds or attempts; "
                             "refusing to mix splits")
        if key in out:
            raise SystemExit(f"{label} repeats {key}")
        out[key] = r
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{label} is incomplete: {len(missing)} of {len(want)} planned episodes missing, "
                         f"e.g. {sorted(missing)[0]}")
    return out


def limits_of(config: dict, task_id: str) -> dict:
    return {"completion_tokens": config["completion_tokens"].get(task_id),
            "context_tokens": config.get("context_tokens", {}).get(task_id)}


def arm_values(b: dict, c: dict, split: dict, plan: dict) -> tuple[dict, dict, dict]:
    """Per-task (seeds x attempts) reward arrays. Which rows are excluded is stats.episode_status, under
    each arm's own limits. For mixed_rate an excluded row is dropped, as the trainer keeps the first 4
    valid episodes of 5; for the reward metrics it counts as a failure (0) for the arm that produced it.
    Arms whose excluded rates differ by more than the plan's error tolerance are refused, so an arm
    cannot win by erroring on the episodes it would fail."""
    excluded = lambda arm, cfg: {k: episode_status(r, limits_of(cfg, k[0])) in  # noqa: E731
                                 ("error", "harness_budget", "under_limit") for k, r in arm.items()}
    flags = {"baseline": excluded(b, plan["baseline"]), "candidate": excluded(c, plan["candidate"])}
    rates = {label: sum(f.values()) / len(f) for label, f in flags.items()}
    tolerance = plan["rule"]["error_tolerance"]
    if abs(rates["baseline"] - rates["candidate"]) > tolerance:
        raise SystemExit(f"error/budget rates differ: baseline {rates['baseline']:.3f}, candidate "
                         f"{rates['candidate']:.3f} (tolerance {tolerance})")
    for key in b:
        if b[key].get("corpus") != c[key].get("corpus"):
            raise SystemExit(f"{key} ran different corpus functions in the two arms")
    stable = lambda r: tuple(r.get(k) for k in STABLE)  # noqa: E731
    if all(stable(b[k]) == stable(c[k]) for k in b):
        raise SystemExit("baseline and candidate are the same runs (identical apart from timings); "
                         "each arm must be its own run")
    dropped = np.nan if plan["metric"] == "mixed_rate" else 0.0

    def arrays(arm, flag):
        out = {t: np.array([[dropped if flag[(t, s, a)] else float(arm[(t, s, a)]["reward"])
                             for a in range(split["attempts"])] for s in split["seeds"]], dtype=float)
               for t in split["task_ids"]}
        short = [t for t, v in out.items() if np.sum(~np.isnan(v)) < plan["group_size"]]
        if short:
            raise SystemExit(f"fewer than {plan['group_size']} valid episodes for {short}")
        return out
    return arrays(b, flags["baseline"]), arrays(c, flags["candidate"]), rates


def task_metric(values: np.ndarray, idx: np.ndarray, metric: str, g: int, codes=None) -> np.ndarray:
    """Metric for each bootstrap row of seed indices ``idx`` (B x seeds)."""
    picked = values[idx]  # B x seeds x attempts
    if metric == "mean_reward":
        return picked.mean(axis=(1, 2))
    if metric == "solve_rate":
        return (picked == 1.0).mean(axis=(1, 2))
    onehot = codes[idx]  # B x seeds x attempts x K; a dropped (excluded) episode is all zeros
    counts = onehot.sum(axis=(1, 2)).astype(float)
    n = counts.sum(axis=1)
    falling = lambda x: np.prod([x - i for i in range(g)], axis=0)  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):  # a resample with fewer than g valid episodes
        same = falling(counts.T).sum(axis=0) / falling(n)
    return np.where(n >= g, 1.0 - same, np.nan)  # unbiased P(g valid draws not all equal)


def weighted(per_task: dict, mix: dict) -> np.ndarray:
    by_level = defaultdict(list)
    for task_id, values in per_task.items():
        by_level[level_of(task_id)].append(values)
    total = 0.0
    for level, weight in mix.items():
        if weight == 0:
            continue
        if level not in by_level:
            raise SystemExit(f"mix weights {level} at {weight} but the split has no {level} cells")
        total = total + weight * np.mean(by_level[level], axis=0)
    return total


def evaluate_split(base, cand, split, plan, rng) -> dict:
    b, c = index_arm(base, split, "baseline"), index_arm(cand, split, "candidate")
    bv, cv, rates = arm_values(b, c, split, plan)
    n_seeds = len(split["seeds"])
    idx = np.vstack([np.arange(n_seeds), rng.integers(n_seeds, size=(plan["rule"]["boot"], n_seeds))])
    per = {"b": {}, "c": {}}
    for t in split["task_ids"]:  # stratified by task, paired by seed: one index matrix for both arms
        codes = None
        if plan["metric"] == "mixed_rate":
            both = np.round(np.concatenate([bv[t], cv[t]]), 9)
            values = np.unique(both[~np.isnan(both)])
            onehot = (both[..., None] == values).astype(float)  # NaN matches nothing
            codes_b, codes_c = onehot[:n_seeds], onehot[n_seeds:]
        for arm, vals in (("b", bv[t]), ("c", cv[t])):
            if plan["metric"] == "mixed_rate":
                codes = codes_b if arm == "b" else codes_c
            per[arm][t] = task_metric(vals, idx, plan["metric"], plan["group_size"], codes)
    delta = weighted(per["c"], plan["candidate"]["difficulty_mix"]) - weighted(per["b"], plan["baseline"]["difficulty_mix"])
    lower = float(np.nanquantile(delta[1:], plan["rule"]["alpha"]))
    upper = float(np.nanquantile(delta[1:], 1.0 - ALPHA))
    return {"episodes": len(b), "excluded_rate": rates, "delta": float(delta[0]), "lower_bound": lower,
            "upper_bound": upper, "bootstrap_90": np.nanquantile(delta[1:], [0.05, 0.95]).tolist()}


def decide(plan: dict, base_disc, cand_disc, base_conf, cand_conf) -> dict:
    if plan.get("kind") != "onecell-plan":
        raise SystemExit("not an onecell plan")
    if plan.get("plan_sha256") != plan_sha(plan):
        raise SystemExit("plan was edited after it was proposed (plan_sha256 mismatch)")
    if changed_knobs(plan["baseline"], plan["candidate"]) != [plan["knob"]]:
        raise SystemExit("plan changes more than one knob")
    if plan["knob"] == "difficulty_mix" and plan["metric"] != "mixed_rate":
        raise SystemExit("difficulty_mix may only be scored by mixed_rate")
    rng = np.random.default_rng(plan["seed"])
    disc = evaluate_split(base_disc, cand_disc, plan["discovery"], plan, rng)
    conf = evaluate_split(base_conf, cand_conf, plan["confirmation"], plan, rng)
    rule = plan["rule"]
    gain = disc["lower_bound"] > rule["margin"]
    no_loss = conf["lower_bound"] > -rule["confirm_tolerance"]
    reasons = []
    if not gain:
        reasons.append(f"discovery lower bound {disc['lower_bound']:+.4f} (delta {disc['delta']:+.4f}) "
                       f"is not > margin {rule['margin']}")
    if not no_loss:
        reasons.append(f"confirmation lower bound {conf['lower_bound']:+.4f} (delta {conf['delta']:+.4f}) "
                       f"does not rule out a loss beyond {rule['confirm_tolerance']}")
    return {"knob": plan["knob"], "from": plan["from"], "to": plan["to"], "metric": plan["metric"],
            "margin": rule["margin"], "retest": plan.get("retest", 0), "alpha": rule["alpha"],
            "accept": gain and no_loss, "discovery": disc, "confirmation": conf,
            "reasons": reasons or ["discovery gain and no confirmation loss, both beyond bootstrap noise"]}


def init_config(template: list[dict]) -> dict:
    tree = ast.parse((ROOT / "qe_arena" / "environment.py").read_text())
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    levels = Counter(level_of(e["task_id"]) for e in template)
    mix = {lv: round(levels.get(lv, 0) / len(template), 6) for lv in LEVELS}
    mix[max(mix, key=mix.get)] += round(1.0 - sum(mix.values()), 6)
    return {"max_steps": consts["MAX_STEPS"], "max_test_runs": consts["MAX_TEST_RUNS"], "difficulty_mix": mix,
            "completion_tokens": {e["task_id"]: e["completion_tokens"] for e in template},
            "context_tokens": {e["task_id"]: e["context_tokens"] for e in template}}
