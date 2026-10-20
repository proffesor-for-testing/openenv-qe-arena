#!/usr/bin/env python3
"""Export the public task dataset: what the model sees at reset, never verifier data.

    python scripts/export_dataset.py OUT_DIR --seeds 64

Writes tasks.jsonl (the software-engineering cells, byte-identical to earlier exports),
domain_tasks.jsonl (the cells scoped to other arena domains), manifest.json and README.md (the
dataset card). Rows are reproducible: the environment rebuilds the same task from (task_id, seed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qe_arena import corpus, skills  # noqa: E402
from qe_arena.tasks import DOMAIN_TASK_IDS, TASK_IDS, make_task, task_domain  # noqa: E402
from qe_arena.terminal_env import TERMINAL_TASK_IDS, instruction  # noqa: E402

CARD = """---
license: mit
language: [en]
task_categories: [text-generation]
tags: [openenv, openenv-arena, reinforcement-learning, software-engineering, quality-engineering, mutation-testing, flaky-tests{domain_tags}]
pretty_name: OpenEnv QE Arena tasks
configs:
  - config_name: default
    data_files: tasks.jsonl
  - config_name: domains
    data_files: domain_tasks.jsonl
  - config_name: terminal
    data_files: terminal_tasks.jsonl
---

# OpenEnv QE Arena tasks

Public task instances for the [openenv-qe-arena](https://github.com/proffesor-for-testing/openenv-qe-arena)
training environment (OpenEnv Arena). The same quality-engineering skills are applied to code from several
arena domains: the `default` config holds the software-engineering cells, and the `domains` config holds the
cells whose corpus is code from another domain ({domain_list}).

| Family | What the model does | Reward |
|---|---|---|
| `qe_bugfix` | Repairs an injected bug by exact-replacement edits, with visible tests it can run | 1.0 only if every hidden differential test passes |
| `qe_bugfix-d4` | The same repair under a total edit budget and a localised-repair rule (no rewrite) | as `qe_bugfix`, and 0 for a file that breaks the rule |
| `qe_mutant_kill` | Writes test cases (as data) that tell a mutant apart from the original | 0.5 per mutant for a distinguishing input, 1.0 when the expected value is also exact |
| `qe_flaky` | Finds the flaky test in a suite (d1), names its cause (d2), or repairs it by edits (d3); some suites are clean | d1: 1.0 for the right test or null; d2: 0.5 for the test, 1.0 with the cause; d3: 0.5 for the test, 1.0 for a repair that holds on every graded run |

Each row is what the model sees at reset: `prompt` and the virtual `files`. Hidden inputs, reference outputs,
mutant sources and flaky-test ground truth are generated inside the environment from `(task_id, seed)` and are not in this dataset.
`qe_bugfix-d4` rows (and their domain variants) also carry `edit_budget` (the budget the prompt states); other
rows have no such key.

A task ID names its domain between the family and the level: `qe_mutant_kill-d2` is the software-engineering
cell, and `qe_mutant_kill-finance-d2` draws only finance-economics code. Every row's `domain` is the arena
domain with spaces for hyphens.

## Terminal interface

The `terminal` config is the same QE work behind a terminal (the `openenv-qe-arena-terminal` image): the
model runs bash commands in a real, isolated working directory (`exec`), writes files (`write_file`) and ends
the episode with `evaluate`, which grades one file with the same deterministic graders: `module.py` for a bug
fix, `answer.json` (`{{"tests": [{{"args": [...], "expected": ...}}]}}`) for mutant-kill. Rewards are 1 or 0:
every hidden test passes, or every mutant is killed with exact expected values. Nothing the shell prints
reaches the reward, and the hidden tests never exist on disk. Each row's `instruction` is the reset
observation and `files` the working directory; bug-fix and mutant-kill d1-d3 in every domain are served.

- Rows: {rows} ({seeds} seeds x {cells} task cells) in `tasks.jsonl`; {domain_rows} ({seeds} seeds x
  {domain_cells} task cells) in `domain_tasks.jsonl`; {terminal_rows} ({seeds} seeds x {terminal_cells} task
  cells) in `terminal_tasks.jsonl`
- Corpus: {entries} small Python functions, {adapted} of them adapted from MIT-licensed QE projects and the rest
  original; origins and domains are in `manifest.json`.
- Source commit: `{commit}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    seeds = range(args.seed_base, args.seed_base + args.seeds)
    rows, domain_rows = export_rows(TASK_IDS, seeds), export_rows(DOMAIN_TASK_IDS, seeds)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    domain_data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in domain_rows)
    (args.out / "tasks.jsonl").write_text(data)
    (args.out / "domain_tasks.jsonl").write_text(domain_data)
    terminal_rows = export_terminal_rows(TERMINAL_TASK_IDS, seeds)
    terminal_data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in terminal_rows)
    (args.out / "terminal_tasks.jsonl").write_text(terminal_data)
    manifest = {
        "rows": len(rows), "task_ids": TASK_IDS, "seeds": [args.seed_base, args.seed_base + args.seeds - 1],
        "tasks_sha256": hashlib.sha256(data.encode()).hexdigest(), "source_commit": commit,
        "domain_rows": len(domain_rows), "domain_task_ids": DOMAIN_TASK_IDS,
        "domain_tasks_sha256": hashlib.sha256(domain_data.encode()).hexdigest(),
        "terminal_rows": len(terminal_rows), "terminal_task_ids": TERMINAL_TASK_IDS,
        "terminal_tasks_sha256": hashlib.sha256(terminal_data.encode()).hexdigest(),
        "corpus": {name: {"entry": e.entry, "origin": e.origin, "domain": e.domain}
                   for name, e in sorted(corpus.ENTRIES.items())},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    domains = sorted({task_domain(t) for t in DOMAIN_TASK_IDS})
    (args.out / "README.md").write_text(CARD.format(
        rows=len(rows), seeds=args.seeds, cells=len(TASK_IDS), domain_rows=len(domain_rows),
        domain_cells=len(DOMAIN_TASK_IDS), terminal_rows=len(terminal_rows), terminal_cells=len(TERMINAL_TASK_IDS),
        domain_list=", ".join(domains) or "none yet",
        domain_tags="".join(f", {d}" for d in domains), entries=len(corpus.ENTRIES),
        adapted=sum(e.origin != "Original." for e in corpus.ENTRIES.values()), commit=commit))
    print(f"wrote {len(rows)} rows to {args.out}/tasks.jsonl (sha256 {manifest['tasks_sha256'][:16]}) and "
          f"{len(domain_rows)} to domain_tasks.jsonl (sha256 {manifest['domain_tasks_sha256'][:16]}) and "
          f"{len(terminal_rows)} to terminal_tasks.jsonl (sha256 {manifest['terminal_tasks_sha256'][:16]})")
    return 0


def domain_label(task_id: str) -> str:
    """The row's `domain`: the arena domain with spaces, which keeps "software engineering" as it was."""
    domain = skills.parse_skill_id(task_id)[0] if task_id in skills.SKILL_TASK_IDS else task_domain(task_id)
    return domain.replace("-", " ")


def export_rows(task_ids: list[str], seeds: range) -> list[dict]:
    rows = []
    for task_id in task_ids:
        for seed in seeds:
            task = make_task(task_id, seed)
            row = {
                "task_id": task_id, "seed": seed, "split": "train", "domain": domain_label(task_id),
                "family": task.family, "difficulty": task.difficulty,
                "prompt": task.prompt, "files": json.dumps(task.files, sort_keys=True),
            }
            if task.private.get("edit_budget") is not None:  # d4 only, so other rows stay byte-identical
                row["edit_budget"] = task.private["edit_budget"]
            rows.append(row)
    return rows


def export_terminal_rows(task_ids: list[str], seeds: range) -> list[dict]:
    """The terminal interface's reset observation (exec available, as in the isolated image) and files."""
    rows = []
    for task_id in task_ids:
        for seed in seeds:
            task = skills.make_skill_task(task_id, seed) if task_id in skills.SKILL_TASK_IDS else make_task(task_id, seed)
            rows.append({
                "task_id": task_id, "seed": seed, "split": "train", "domain": domain_label(task_id),
                "family": task.family, "difficulty": task.difficulty, "interface": "terminal",
                "instruction": instruction(task, sorted(task.files), True),
                "files": json.dumps(task.files, sort_keys=True),
            })
    return rows


if __name__ == "__main__":
    sys.exit(main())
