"""Thompson-sample task cells that give GRPO a gradient."""

from __future__ import annotations

import copy

import numpy as np

from .stats import TRAIN_GROUP, expected_all_equal

OBJECTIVES = {
    "projected": "p_mix@G: chance that G fresh instances do not all get the same reward (partial credit "
                 "counts), drawn from a Dirichlet posterior over the observed reward values",
    "group": "same-seed informative-group posterior (measured; understates mixing for unseeded resets)",
}
# Gate: posterior-mean p_mix@G below this means the cell rarely gives a gradient. With Jeffreys pseudo-
# counts at G=4 that excludes 0/16 or 16/16 (0.10) but keeps 0/8 (0.18, still worth exploring) and
# 15/16 (0.29); Thompson sampling then ranks what remains.
MIN_MIXED = 0.15


def select_cells(stats: dict, template: list[dict], max_cells: int, seed: int, min_mixed: float = MIN_MIXED,
                 min_groups: int = 1, min_episodes: int = 4, objective: str = "projected",
                 group_size: int = TRAIN_GROUP) -> tuple[list[dict], list[dict]]:
    """Thompson-sample eligible cells by ``objective`` (see OBJECTIVES). Returns the task list
    (template entries copied verbatim, in template order) and one reason record per cell."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
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
            mixed = 1.0 - expected_all_equal(s["reward_dirichlet"]["alphas"], group_size)
            record.update(solve=s["solve"], informative=s["informative"], errors=s["errors"], budget=s["budget"],
                          valid=s["valid"], mean_reward=s["mean_reward"], p_mixed=float(mixed),
                          reward_dirichlet=s["reward_dirichlet"])
        if s is None:
            record.update(decision="excluded", reason="no calibration data")
        elif task_id not in by_id:
            record.update(decision="excluded", reason="no template entry, so its limits are unknown")
        elif objective == "group" and s["groups"] < min_groups:
            record.update(decision="excluded", reason=f"{s['groups']} same-seed groups < {min_groups}")
        elif s["valid"] < min_episodes:
            record.update(decision="excluded", reason=f"{s['valid']} valid episodes < {min_episodes}")
        elif record["p_mixed"] < min_mixed:
            side = "mostly high reward" if (s["mean_reward"] or 0) >= 0.5 else "mostly low reward"
            record.update(decision="excluded", reason=f"rarely mixed ({side}): posterior-mean p_mix@{group_size} "
                                                      f"{record['p_mixed']:.3f} < {min_mixed}")
        else:
            eligible.append(record)
        reasons.append(record)
    rng = np.random.default_rng(seed)
    for record in eligible:  # sorted order, so one seed gives one draw sequence
        if objective == "group":
            post = record["informative"]
            record["draw"] = float(rng.beta(post["a"], post["b"]))
        else:
            q = rng.dirichlet(record["reward_dirichlet"]["alphas"])
            record["draw"] = float(1.0 - np.sum(q ** group_size))
    ranked = sorted(eligible, key=lambda r: (-r["draw"], r["task_id"]))
    measure = f"p_mix@{group_size}" if objective == "projected" else "same-seed informative"
    for rank, record in enumerate(ranked, 1):
        record["rank"] = rank
        chosen = rank <= max_cells
        record["decision"] = "selected" if chosen else "excluded"
        record["reason"] = (f"Thompson draw {record['draw']:.3f} ({measure}) ranked {rank} of {len(ranked)} eligible"
                            + ("" if chosen else f"; beyond --max-cells {max_cells}"))
    chosen_ids = {r["task_id"] for r in ranked[:max_cells]}
    tasks = [copy.deepcopy(e) for e in template if e["task_id"] in chosen_ids]
    return tasks, reasons
