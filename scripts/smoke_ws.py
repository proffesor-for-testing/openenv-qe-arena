#!/usr/bin/env python3
"""Drive a running server over the native OpenEnv WebSocket protocol.

For every task ID and a few seeds:
  1. the declared example actions end the episode with reward 0;
  2. the oracle solution earns exactly 1.0;
  3. the same seed reproduces the same reset observation;
  4. no observation carries private verifier keys.

    python scripts/smoke_ws.py --url ws://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from oracle import solve_actions, terminal_solve_actions  # noqa: E402
from qe_arena import skills  # noqa: E402
from qe_arena.tasks import ALL_TASK_IDS, make_task  # noqa: E402

TERMINAL_EXAMPLES = [{"action_type": "evaluate"}]

FORBIDDEN = {"expected", "private", "original", "reference", "mutants", "seed", "inputs"}


def exchange(ws, kind: str, data: dict) -> dict:
    ws.send(json.dumps({"type": kind, "data": data}))
    message = json.loads(ws.recv(timeout=60))
    assert message.get("type") == "observation", message
    result = message["data"]
    reward = result.get("reward")
    assert isinstance(reward, (int, float)) and math.isfinite(reward) and 0 <= reward <= 1, result
    leaked = FORBIDDEN.intersection(result.get("observation", {}))
    assert not leaked, f"private keys in observation: {leaked}"
    return result


def episode(url: str, task_id: str, seed: int, actions: list[dict]) -> tuple[dict, dict]:
    with connect(url.rstrip("/") + "/ws", max_size=2**22) as ws:
        reset = exchange(ws, "reset", {"task_id": task_id, "seed": seed})
        last = reset
        for action in actions:
            last = exchange(ws, "step", action)
            if last["done"]:
                break
        return reset, last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="ws://host:port")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--terminal", action="store_true", help="the terminal-interface server (Dockerfile.terminal)")
    args = parser.parse_args()
    if args.terminal:
        from qe_arena.terminal_env import TERMINAL_TASK_IDS  # noqa: PLC0415

        task_ids, examples, solve = TERMINAL_TASK_IDS, TERMINAL_EXAMPLES, terminal_solve_actions
    else:
        task_ids, solve = ALL_TASK_IDS, solve_actions
        examples = json.loads((ROOT / "example-actions.json").read_text())
    failures = 0
    for task_id in task_ids:
        for seed in range(9_000_001, 9_000_001 + args.seeds):
            checks = {}
            reset_a, last = episode(args.url, task_id, seed, examples)
            checks["example_reward_0"] = last["done"] and last["reward"] == 0
            build = skills.make_skill_task if task_id in skills.ALL_SKILL_IDS else make_task
            reset_b, last = episode(args.url, task_id, seed, solve(build(task_id, seed)))
            checks["oracle_reward_1"] = last["done"] and last["reward"] == 1
            checks["seeded_reset_replay"] = reset_a["observation"] == reset_b["observation"]
            ok = all(checks.values())
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {task_id} seed={seed} {checks}")
    print(f"{len(task_ids) * args.seeds - failures}/{len(task_ids) * args.seeds} task episodes passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
