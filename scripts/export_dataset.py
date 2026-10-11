#!/usr/bin/env python3
"""Export the public task dataset: what the model sees at reset, never verifier data.

    python scripts/export_dataset.py OUT_DIR --seeds 64

Writes tasks.jsonl, manifest.json and README.md (the dataset card). Rows are
reproducible: the environment rebuilds the same task from (task_id, seed).
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

from qe_arena import corpus  # noqa: E402
from qe_arena.tasks import DOMAIN, TASK_IDS, make_task  # noqa: E402

CARD = """---
license: mit
language: [en]
task_categories: [text-generation]
tags: [openenv, openenv-arena, reinforcement-learning, software-engineering, quality-engineering, mutation-testing]
pretty_name: OpenEnv QE Arena tasks
configs:
  - config_name: default
    data_files: tasks.jsonl
---

# OpenEnv QE Arena tasks

Public task instances for the [openenv-qe-arena](https://github.com/proffesor-for-testing/openenv-qe-arena)
training environment (OpenEnv Arena, software-engineering domain).

| Family | What the model does | Reward |
|---|---|---|
| `qe_bugfix` | Repairs an injected bug by exact-replacement edits, with visible tests it can run | 1.0 only if every hidden differential test passes |
| `qe_mutant_kill` | Writes test cases (as data) that tell a mutant apart from the original | 0.5 per mutant for a distinguishing input, 1.0 when the expected value is also exact |

Each row is what the model sees at reset: `prompt` and the virtual `files`. Hidden inputs, reference outputs
and mutant sources are generated inside the environment from `(task_id, seed)` and are not in this dataset.

- Rows: {rows} ({seeds} seeds x {cells} task cells)
- Corpus: {entries} small Python functions, most adapted from MIT-licensed QE projects; origins are in
  `manifest.json`.
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
    rows = []
    for task_id in TASK_IDS:
        for seed in range(args.seed_base, args.seed_base + args.seeds):
            task = make_task(task_id, seed)
            rows.append({
                "task_id": task_id, "seed": seed, "split": "train", "domain": DOMAIN,
                "family": task.family, "difficulty": task.difficulty,
                "prompt": task.prompt, "files": json.dumps(task.files, sort_keys=True),
            })
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (args.out / "tasks.jsonl").write_text(data)
    manifest = {
        "rows": len(rows), "task_ids": TASK_IDS, "seeds": [args.seed_base, args.seed_base + args.seeds - 1],
        "tasks_sha256": hashlib.sha256(data.encode()).hexdigest(), "source_commit": commit,
        "corpus": {name: {"entry": e.entry, "origin": e.origin} for name, e in sorted(corpus.ENTRIES.items())},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.out / "README.md").write_text(CARD.format(rows=len(rows), seeds=args.seeds, cells=len(TASK_IDS),
                                                     entries=len(corpus.ENTRIES), commit=commit))
    print(f"wrote {len(rows)} rows to {args.out} (sha256 {manifest['tasks_sha256'][:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
