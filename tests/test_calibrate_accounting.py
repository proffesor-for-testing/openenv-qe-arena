"""calibrate.py scores episodes by the arena guide's rules (AGENTS.md "Replies" and "Budgets"):
non-terminal ends go through finish_action, and completion_tokens counts observations after the first."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate  # noqa: E402

READ_ALL = '{"op": "read", "path": "*"}'


def run(monkeypatch, replies, *, strict=False, max_steps=16, completion=4096, context=8192,
        finish=0.5, ceiling=10**6):
    """Play scripted replies; each is (text, prompt_tokens, completion_tokens). finish_reward is stubbed."""
    calls = []

    def chat(args, token, messages):
        text, prompt, generated = replies[min(len(calls), len(replies) - 1)]
        calls.append(len(messages))
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": generated,
                          "total_tokens": prompt + generated}}

    monkeypatch.setattr(calibrate, "chat", chat)
    monkeypatch.setattr(calibrate, "finish_reward", lambda env: finish)
    args = SimpleNamespace(fault_kind=None, episode_completion_tokens=completion, episode_context_tokens=context,
                           max_steps=max_steps, strict=strict)
    record = calibrate.run_episode(args, "", calibrate.Budget(ceiling), "qe_flaky-d1", 7, 0)
    return record, calls


def test_the_step_cap_is_scored_through_finish_action_not_zero(monkeypatch):
    record, calls = run(monkeypatch, [(READ_ALL, 100, 10)], max_steps=3)
    assert record["end"] == "step_cap" and record["end_finish"] and len(calls) == 3
    assert record["reward"] == record["lenient_reward"] == 0.5


def test_completion_budget_counts_observations_after_the_first(monkeypatch):
    # The model generates only 10 tokens a turn, but the observations push the conversation past 4096
    # tokens beyond the first prompt; counting generated tokens alone would never end this episode.
    replies = [(READ_ALL, 100, 10), (READ_ALL, 2000, 10), (READ_ALL, 4300, 10)]
    record, calls = run(monkeypatch, replies)
    assert record["generated_tokens"] == 30 and record["completion_tokens"] == 4300 + 10 - 100
    assert record["end"] == "completion_budget" and record["end_finish"] and record["reward"] == 0.5
    assert record["steps"] == 2  # the reply that overflowed was never applied


def test_context_budget_covers_the_whole_conversation(monkeypatch):
    replies = [(READ_ALL, 3000, 10), (READ_ALL, 8100, 200)]
    record, _ = run(monkeypatch, replies, completion=10**6)
    assert record["end"] == "context_budget" and record["context_tokens_max"] == 8300
    assert record["end_finish"] and record["reward"] == 0.5 and record["steps"] == 1


def test_a_strict_end_is_scored_through_finish_action_at_that_point(monkeypatch):
    replies = [(READ_ALL, 100, 10), ("Let me look again.\n" + READ_ALL, 150, 10)]
    record, _ = run(monkeypatch, replies, strict=True, max_steps=4, finish=0.25)
    assert record["strict_end"] and record["strict_end_step"] == 2 and record["finish_reward"] == 0.25
    assert record["reward"] == 0.25  # was forced to 0 before; the arena grades it through finish_action
    assert record["end"] == "step_cap" and record["lenient_reward"] == 0.25


def test_our_own_token_ceiling_is_not_an_arena_outcome(monkeypatch):
    record, _ = run(monkeypatch, [(READ_ALL, 100, 10)], ceiling=50)
    assert record["end"] == "global_budget" and record["reward"] is None and not record["end_finish"]


def test_a_submit_is_scored_by_the_environment_not_finish_action(monkeypatch):
    submit = json.dumps({"op": "submit", "answer": {"test": "nope"}})
    record, _ = run(monkeypatch, [(READ_ALL, 100, 10), (submit, 150, 10)], finish=0.9)
    assert record["end"] == "submit" and not record["end_finish"] and record["reward"] == 0.0

