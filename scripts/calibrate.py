#!/usr/bin/env python3
"""Measure how the arena's model does on each task cell, in the trainer's likely regime.

Runs Qwen3.8-27B through an OpenAI-compatible endpoint (default: the HF router)
with thinking OFF against the environment in-process, and writes one JSON line
per episode with the full trajectory, token usage and reward. Dry run unless
--execute is given. The HF token is read from the local HF cache inside this
process; it never appears in argv, output or logs.

    python scripts/calibrate.py --task-id qe_bugfix-d1 --attempts 4 --seeds 2 \
        --output runs/cal.jsonl --max-total-tokens 2000000 --execute

Episodes are scored as the arena's guide says (AGENTS.md, "Replies" and "Budgets"):

- An episode that does not end by itself (step cap, completion or context budget) is still scored, through
  finish_action ({"op": "submit", "answer": {}}): a bug fix keeps its edits. ``end_finish`` marks those rows.
- ``completion_tokens`` covers the model's tokens and every observation after the first, so it is measured from
  the API's usage as (prompt + completion) - first prompt; ``context_tokens`` covers the whole conversation.
  A reply or observation that pushes either over its budget ends the episode before that reply counts.
- --interface terminal drives the terminal environment (qe_arena/terminal_env.py) instead: actions are exec /
  write_file / evaluate, finish_action is {"action_type": "evaluate"} and the step cap defaults to its 32. Its
  commands run in a real shell, so this refuses to start unless the shell is uid-isolated, i.e. inside the
  terminal image as root; see docs/DESIGN.md, "Terminal interface".
- --strict: a reply that is not exactly one JSON object (json.loads of the stripped reply must give a dict;
  prose, fences or a second object fail) ends the episode, which is then scored through finish_action at that
  point. The episode still plays on under the lenient parser, so each row records both: ``reward`` (strict),
  ``lenient_reward``, ``strict_end``, ``strict_end_step`` and ``finish_reward`` (the strict-end score).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qe_arena import shell  # noqa: E402
from qe_arena.environment import QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import DOMAIN_TASK_IDS, TASK_IDS, cell_names, parse_task_id  # noqa: E402
from qe_arena.terminal_env import MAX_STEPS as TERMINAL_STEPS  # noqa: E402
from qe_arena.terminal_env import TERMINAL_TASK_IDS, TerminalAction, TerminalEnvironment  # noqa: E402

# Names, resolved per episode, so a test can substitute the environment class.
INTERFACES = {"json": ("QEEnvironment", "QEAction", "op", "submit"),
              "terminal": ("TerminalEnvironment", "TerminalAction", "action_type", "evaluate")}

SYSTEM = ("Solve the task with the available tools. Each reply must be exactly one JSON action object, "
          "with no markdown. Action schema: ")


def hf_token() -> str:
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    token = path.read_text().strip()
    if not token:
        raise SystemExit("no HF token in the local cache; run `hf auth login`")
    return token


class Budget:
    def __init__(self, limit: int):
        self.limit, self.used, self.lock = limit, 0, Lock()

    def charge(self, tokens: int) -> bool:
        with self.lock:
            self.used += tokens
            return self.used <= self.limit


def is_hf_router(base_url: str) -> bool:
    return "router.huggingface.co" in base_url


def chat(args, token: str, messages: list[dict]) -> dict:
    payload = {
        "model": args.model, "messages": messages, "temperature": args.temperature,
        "max_tokens": args.max_tokens, "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
    }
    if args.thinking == "off" and not is_hf_router(args.base_url):
        payload["reasoning_effort"] = "none"  # Ollama ignores chat_template_kwargs; this turns thinking off
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            body = error.read()[:300].decode(errors="replace").replace(token, "[redacted]")
            if error.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"HTTP {error.code}: {body}") from None
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt < 3:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"network: {error}") from None
    raise RuntimeError("unreachable")


def parse_action(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def parse_strict(text: str) -> dict | None:
    """The arena's rule: the stripped reply is exactly one JSON object. No fences, no prose, no second object."""
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def finish_reward(env) -> float:
    """What finish_action would score now, without changing the episode, so the lenient episode plays on."""
    if isinstance(env, TerminalEnvironment):
        return env.peek_reward()  # a deep copy would share, and on evaluate delete, the live workspace
    probe = copy.deepcopy(env)
    obs = probe.step(QEAction(op="submit", answer={}))
    return float(obs.reward or 0.0)


def observation_text(obs) -> str:
    data = {k: v for k, v in obs.model_dump().items() if k not in ("metadata",) and v not in ("", [], None)}
    return json.dumps(data, ensure_ascii=False)


def run_episode(args, token: str, budget: Budget, task_id: str, seed: int, attempt: int,
                corpus_name: str | None = None) -> dict:
    env_name, action_name, op_field, finish_op = INTERFACES[getattr(args, "interface", "json")]
    env_class, action_class = globals()[env_name], globals()[action_name]
    env = env_class(corpus_pool=(corpus_name,) if corpus_name else None, fault_kind=args.fault_kind)
    try:
        return _play(args, token, budget, task_id, seed, attempt, env, action_class, op_field, finish_op)
    finally:
        if hasattr(env, "close"):
            env.close()


def _play(args, token, budget, task_id, seed, attempt, env, action_class, op_field, finish_op) -> dict:
    started = time.time()
    obs = env.reset(seed=seed, task_id=task_id)
    schema = json.dumps(action_class.model_json_schema(), separators=(",", ":"))
    messages = [{"role": "system", "content": SYSTEM + schema},
                {"role": "user", "content": observation_text(obs)}]
    record = {"task_id": task_id, "seed": seed, "attempt": attempt, "corpus": env._task.private["corpus"],
              "fault_kind": env._task.private.get("fault_kind"),
              "faults": [f.get("operator") or f.get("id") for f in env._task.private.get("faults", [])],
              "steps": 0, "invalid": 0, "nonstrict": 0, "strict": args.strict,
              "strict_end": False, "strict_end_step": None, "finish_reward": None,
              "episode_completion_tokens": args.episode_completion_tokens,
              "episode_context_tokens": args.episode_context_tokens, "max_steps": args.max_steps,
              "completion_tokens": 0, "generated_tokens": 0, "first_prompt_tokens": None,
              "prompt_tokens_last": 0, "prompt_tokens_max": 0, "context_tokens_max": 0,
              "end": "", "end_finish": False, "reward": 0.0}
    while True:
        reply = chat(args, token, messages)
        usage = reply.get("usage") or {}
        prompt_tokens, generated = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        if record["first_prompt_tokens"] is None:
            record["first_prompt_tokens"] = prompt_tokens
        record["generated_tokens"] += generated
        record["prompt_tokens_last"] = prompt_tokens
        record["prompt_tokens_max"] = max(record["prompt_tokens_max"], prompt_tokens)
        if not budget.charge(int(usage.get("total_tokens", 0))):
            record["end"], record["reward"] = "global_budget", None  # our own ceiling, not an arena outcome
            break
        conversation = prompt_tokens + generated
        record["context_tokens_max"] = max(record["context_tokens_max"], conversation)
        # The arena's completion budget: the model's tokens plus every observation after the first.
        record["completion_tokens"] = conversation - record["first_prompt_tokens"]
        if conversation > args.episode_context_tokens:
            record["end"] = "context_budget"
            break
        if record["completion_tokens"] > args.episode_completion_tokens:
            record["end"] = "completion_budget"
            break
        text = (reply["choices"][0]["message"].get("content") or "")
        messages.append({"role": "assistant", "content": text})
        action = parse_action(text)
        record["steps"] += 1
        if parse_strict(text) is None:
            record["nonstrict"] += 1
            if not record["strict_end"]:
                record["strict_end"], record["strict_end_step"] = True, record["steps"]
                record["finish_reward"] = finish_reward(env)
        if action is None:
            record["invalid"] += 1
            obs = None
            obs_text = json.dumps({"error": "invalid action: reply with exactly one JSON action object"})
        else:
            try:
                obs = env.step(action_class(**action))
                obs_text = observation_text(obs)
            except Exception as error:  # pydantic validation of a malformed action
                record["invalid"] += 1
                obs_text = json.dumps({"error": f"invalid action: {str(error)[:200]}"})
                obs = None
        if obs is not None and obs.done:
            record["reward"] = float(obs.reward or 0.0)
            record["end"] = "submit" if action and action.get(op_field) == finish_op else "terminal"
            break
        if record["steps"] >= args.max_steps:
            record["end"] = "step_cap"
            break
        messages.append({"role": "user", "content": obs_text})
    if record["end"] in ("step_cap", "context_budget", "completion_budget"):
        record["reward"], record["end_finish"] = finish_reward(env), True  # the arena still scores it
    record["lenient_reward"] = record["reward"]
    if args.strict and record["strict_end"]:
        record["reward"] = record["finish_reward"]  # the episode ended there; finish_action scored it
    record["seconds"] = round(time.time() - started, 1)
    record["messages"] = messages
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="https://router.huggingface.co/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B:novita")
    parser.add_argument("--thinking", choices=["on", "off"], default="off")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024, help="per reply")
    parser.add_argument("--interface", choices=sorted(INTERFACES), default="json")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="tool calls per episode (tool_calls_total); default 16, or 32 for the terminal")
    parser.add_argument("--episode-completion-tokens", type=int, default=4096)
    parser.add_argument("--episode-context-tokens", type=int, default=8192)
    parser.add_argument("--task-id", action="append",
                        help="repeatable; default the software-engineering cells (plus --domain-cells)")
    parser.add_argument("--domain-cells", action="store_true",
                        help="add every domain-scoped cell (e.g. qe_mutant_kill-finance-d2) to the task list")
    parser.add_argument("--seeds", type=int, default=2, help="distinct seeds per task")
    parser.add_argument("--seed-base", type=int, default=500_000)
    parser.add_argument("--attempts", type=int, default=4, help="episodes per seed (GRPO group size)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-total-tokens", type=int, required=True, help="hard global token ceiling")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--each-corpus", action="store_true",
                        help="pin every episode group to one corpus function, for every function of the "
                             "task's domain (difficulty table)")
    parser.add_argument("--fault-kind", choices=["injected", "real"], default=None,
                        help="force the qe_bugfix fault source (default: the environment's own mix)")
    parser.add_argument("--strict", action="store_true",
                        help="score as the arena: a reply that is not exactly one JSON object ends the episode, "
                             "which finish_action then scores")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    terminal = args.interface == "terminal"
    if args.max_steps is None:
        args.max_steps = TERMINAL_STEPS if terminal else 16
    tasks = (args.task_id or TASK_IDS) + (DOMAIN_TASK_IDS if args.domain_cells else [])
    if terminal and not args.task_id:
        tasks = [t for t in tasks if t in TERMINAL_TASK_IDS]
    for task_id in tasks:
        parse_task_id(task_id)
        if terminal and task_id not in TERMINAL_TASK_IDS:
            parser.error(f"{task_id} is not served by the terminal interface")
    if len(set(tasks)) != len(tasks):
        parser.error("a task is listed twice")
    names = {t: list(cell_names(t)) if args.each_corpus else [None] for t in tasks}  # a pool must match the cell
    jobs = [(t, args.seed_base + s, a, n) for t in tasks for n in names[t]
            for s in range(args.seeds) for a in range(args.attempts)]
    pools = sum(len(v) for v in names.values())
    print(f"{len(jobs)} episodes: {len(tasks)} tasks, {pools} task pools, {args.seeds} seeds x {args.attempts} attempts; "
          f"model {args.model}, thinking {args.thinking}, token ceiling {args.max_total_tokens}, "
          f"{'strict' if args.strict else 'lenient'} replies")
    if not args.execute:
        print("dry run; add --execute to call the model")
        return 0
    if terminal and shell.mode() != "uid":
        raise SystemExit(f"--interface terminal runs the model's commands in a real shell; shell mode is "
                         f"{shell.mode()!r}, not 'uid'. Run it inside the terminal image as root.")
    token = hf_token() if is_hf_router(args.base_url) else os.environ.get("OPENAI_API_KEY", "local")
    budget = Budget(args.max_total_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results, lock = [], Lock()
    with args.output.open("w") as out, ThreadPoolExecutor(args.concurrency) as pool:
        def job(spec):
            try:
                record = run_episode(args, token, budget, *spec)
            except Exception as error:
                record = {"task_id": spec[0], "seed": spec[1], "attempt": spec[2], "corpus": spec[3], "end": "error",
                          "error": str(error).replace(token, "[redacted]")[:300], "reward": None}
            with lock:
                out.write(json.dumps(record) + "\n")
                out.flush()
                results.append(record)
        list(pool.map(job, jobs))
    print(f"\ntokens used: {budget.used}")
    if args.each_corpus:
        print(f"{'task':<16}{'corpus':<28} solved  mean")
        for task_id in tasks:
            for name in names[task_id]:
                cell = [r["reward"] for r in results if r["task_id"] == task_id and r.get("corpus") == name
                        and r["reward"] is not None]
                if cell:
                    print(f"{task_id:<16}{name:<28} {sum(x == 1.0 for x in cell)}/{len(cell)}   {statistics.mean(cell):.2f}")
        return 0
    print(f"{'cell':<22}{'seed':>9}  rewards        solved  invalid  nonstrict  ends")
    for task_id in tasks:
        for s in range(args.seeds):
            seed = args.seed_base + s
            rows = sorted((r for r in results if r["task_id"] == task_id and r["seed"] == seed),
                          key=lambda r: r["attempt"])
            rewards = [r["reward"] for r in rows]
            solved = sum(1 for r in rewards if r == 1.0)
            ends = ",".join(sorted({r.get("end", "") for r in rows}))
            print(f"{task_id:<22}{seed:>9}  {str([None if x is None else round(x, 2) for x in rewards]):<15}"
                  f"{solved}/{len(rows)}   {sum(r.get('invalid', 0) for r in rows):>5}  "
                  f"{sum(r.get('nonstrict', 0) for r in rows):>9}    {ends}")
        cell = [r["reward"] for r in results if r["task_id"] == task_id and r["reward"] is not None]
        if cell:
            print(f"{'':<22}{'mean':>9}  {statistics.mean(cell):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
