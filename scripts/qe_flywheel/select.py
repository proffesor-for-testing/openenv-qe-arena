"""Thompson-sample task cells that give GRPO a gradient."""

from __future__ import annotations

import copy

import numpy as np

from .stats import TRAIN_GROUP

OBJECTIVES = {
    "projected": "p_inf@G: chance that G independent instances are mixed, p drawn from the solve posterior",
    "group": "same-seed informative-group posterior (measured; understates mixing for unseeded resets)",
}


def select_cells(stats: dict, template: list[dict], max_cells: int, seed: int, saturated: float = 0.9,
                 dead: float = 0.05, min_groups: int = 1, objective: str = "projected",
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
            record.update(solve=s["solve"], informative=s["informative"], errors=s["errors"],
                          projected_informative=s["projected_informative"])
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
        else:
            p = float(rng.beta(record["solve"]["a"], record["solve"]["b"]))
            record["draw"] = 1.0 - p ** group_size - (1.0 - p) ** group_size
    ranked = sorted(eligible, key=lambda r: (-r["draw"], r["task_id"]))
    for rank, record in enumerate(ranked, 1):
        record["rank"] = rank
        chosen = rank <= max_cells
        record["decision"] = "selected" if chosen else "excluded"
        measure = f"p_inf@{group_size}" if objective == "projected" else "same-seed informative"
        record["reason"] = (f"Thompson draw {record['draw']:.3f} ({measure}) ranked {rank} of {len(ranked)} eligible"
                            + ("" if chosen else f"; beyond --max-cells {max_cells}"))
    chosen_ids = {r["task_id"] for r in ranked[:max_cells]}
    tasks = [copy.deepcopy(e) for e in template if e["task_id"] in chosen_ids]
    return tasks, reasons
