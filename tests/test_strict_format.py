"""The reply-format reminder on step observations, and calibrate.py's strict (arena) reply rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate  # noqa: E402
from qe_arena.environment import STEP_REMINDER, QEAction, QEEnvironment, QEObservation  # noqa: E402
from qe_arena.tasks import make_task  # noqa: E402

READ_ALL = '{"op": "read", "path": "*"}'


@pytest.mark.parametrize("task_id", ["qe_mutant_kill-d1", "qe_flaky-d2"])
def test_every_step_observation_ends_with_the_reminder_and_reset_does_not(task_id):
    env = QEEnvironment()
    obs = env.reset(seed=11, task_id=task_id)
    assert obs.reply_format == "" and obs.prompt == make_task(task_id, 11).prompt
    actions = [{"op": "read", "path": "*"}, {"op": "read", "path": "module.py"}, {"op": "read", "path": "nope.py"},
               {"op": "edit", "path": "module.py", "old": "def ", "new": "def "}]
    for action in actions:
        obs = env.step(QEAction(**action))
        assert not obs.done and obs.reply_format == STEP_REMINDER
        assert list(obs.model_dump())[-1] == "reply_format"  # it closes the rendered observation
    obs = env.step(QEAction(op="submit", answer={}))
    assert obs.done and obs.reply_format == ""  # nothing follows a terminal observation
    assert env.step(QEAction(op="read", path="*")).reply_format == ""


def test_the_reminder_is_one_short_sentence_and_the_schema_says_where_it_goes():
    assert STEP_REMINDER.startswith("Reply with exactly one JSON action object") and len(STEP_REMINDER) < 80
    assert list(QEObservation.model_json_schema()["properties"])[-1] == "reply_format"


@pytest.mark.parametrize("text", [READ_ALL, f"  {READ_ALL}\n", '{"op":"submit","answer":{}}'])
def test_strict_parser_accepts_exactly_one_json_object(text):
    assert calibrate.parse_strict(text) == json.loads(text)


@pytest.mark.parametrize("text", [
    f"```json\n{READ_ALL}\n```",  # fenced
    f"I will read every file first.\n{READ_ALL}",  # prose before
    f"{READ_ALL}\nThat shows the files.",  # prose after
    f"{READ_ALL}\n{READ_ALL}",  # two objects
    '[{"op": "read", "path": "*"}]', '"read"', "", "null",
])
def test_strict_parser_rejects_what_the_lenient_one_forgives(text):
    assert calibrate.parse_strict(text) is None


def test_the_lenient_parser_still_forgives_fences_and_prose():
    for text in (f"```json\n{READ_ALL}\n```", f"I will read every file first.\n{READ_ALL}"):
        assert calibrate.parse_action(text) == json.loads(READ_ALL)


def run_scripted(monkeypatch, strict: bool, prose_before_submit: bool):
    """A flaky-d1 episode: read *, then submit the right answer, optionally with prose before the JSON."""
    made = []

    class Env(QEEnvironment):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    def chat(args, token, messages):
        if len(messages) == 2:
            text = READ_ALL
        else:
            text = json.dumps({"op": "submit", "answer": {"test": made[0]._task.private["victim"]}})
            if prose_before_submit:
                text = "The flaky test is clear now.\n" + text
        return {"choices": [{"message": {"content": text}}],
                "usage": {"completion_tokens": 10, "prompt_tokens": 100, "total_tokens": 110}}

    monkeypatch.setattr(calibrate, "QEEnvironment", Env)
    monkeypatch.setattr(calibrate, "chat", chat)
    args = SimpleNamespace(fault_kind=None, episode_completion_tokens=6144, episode_context_tokens=12288,
                           max_steps=16, strict=strict)
    return calibrate.run_episode(args, "", calibrate.Budget(10**6), "qe_flaky-d1", 7, 0)


@pytest.mark.parametrize("strict", [False, True])
def test_a_prose_reply_ends_a_strict_episode_with_zero_and_keeps_the_lenient_reward(monkeypatch, strict):
    record = run_scripted(monkeypatch, strict, prose_before_submit=True)
    assert record["lenient_reward"] == 1.0 and record["end"] == "submit"
    assert record["strict_end"] and record["strict_end_step"] == 2 and record["nonstrict"] == 1
    assert record["finish_reward"] == 0.0  # finish_action submits {}, which names no test
    assert record["reward"] == (0.0 if strict else 1.0) and record["strict"] is strict


def test_a_clean_episode_scores_the_same_under_both_rules(monkeypatch):
    record = run_scripted(monkeypatch, True, prose_before_submit=False)
    assert record["reward"] == record["lenient_reward"] == 1.0
    assert not record["strict_end"] and record["nonstrict"] == 0 and record["finish_reward"] is None
    assert (record["episode_completion_tokens"], record["episode_context_tokens"]) == (6144, 12288)
    assert all(json.loads(m["content"]).get("reply_format") == STEP_REMINDER
               for m in record["messages"][3:] if m["role"] == "user")
