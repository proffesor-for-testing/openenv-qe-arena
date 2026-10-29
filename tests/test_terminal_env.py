"""The terminal interface: the reward comes only from the graded file, graded by the WASM graders.

These run the shell in ``unsandboxed`` mode (the test's own user, no isolation) or ``disabled`` mode. The uid
isolation itself is exercised by tests/test_shell_isolation.py, which needs root on Linux (the image).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import kill_tests, terminal_solve_actions  # noqa: E402
from qe_arena import shell, terminal_env  # noqa: E402
from qe_arena.sandbox import InfraError  # noqa: E402
from qe_arena.tasks import MODULE, make_task  # noqa: E402
from qe_arena.terminal_env import MAX_STEPS, TERMINAL_TASK_IDS, TerminalAction, TerminalEnvironment  # noqa: E402

CELLS = ["qe_bugfix-d1", "qe_bugfix-d3", "qe_mutant_kill-d1", "qe_mutant_kill-d3", "qe_bugfix-finance-d2",
         "qe_mutant_kill-industrial-d2"]
SEEDS = [11, 2026]
PRIVATE_MARKERS = ("reference", "mutants", "private", "original")


@pytest.fixture(autouse=True)
def unsandboxed(monkeypatch, tmp_path):
    monkeypatch.setenv(shell.ENV_FLAG, "1")
    monkeypatch.setenv(shell.ROOT_ENV, str(tmp_path / "ws"))
    shell._reset_mode_for_tests()
    yield
    shell._reset_mode_for_tests()


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.delenv(shell.ENV_FLAG)
    monkeypatch.setattr(shell, "_probe", lambda: "disabled")
    shell._reset_mode_for_tests()


def act(**fields) -> TerminalAction:
    return TerminalAction(**fields)


def play(task_id: str, seed: int, actions: list[dict]):
    env = TerminalEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    for action in actions:
        obs = env.step(act(**action))
        if obs.done:
            break
    return env, obs


def test_served_cells_are_bugfix_d1_to_d3_and_mutant_kill_in_every_domain():
    families = {t.split("-")[0] for t in TERMINAL_TASK_IDS}
    assert families == {"qe_bugfix", "qe_mutant_kill", "qe_skill"}
    # The only level-4 cells served are the calibrated v9 desk cells; bugfix and mutant-kill stop at d3.
    assert [t for t in TERMINAL_TASK_IDS if t.endswith("-d4")] == [
        "qe_skill-science-d4", "qe_skill-math-d4", "qe_skill-industrial-d4"]
    with pytest.raises(ValueError):
        TerminalEnvironment().reset(seed=1, task_id="qe_flaky-d1")
    with pytest.raises(ValueError):
        TerminalEnvironment().reset(seed=1, task_id="qe_bugfix-d4")


@pytest.mark.parametrize("task_id", CELLS)
@pytest.mark.parametrize("seed", SEEDS)
def test_oracle_earns_exactly_one(task_id, seed):
    _, obs = play(task_id, seed, terminal_solve_actions(make_task(task_id, seed)))
    assert obs.done and obs.reward == 1.0 and obs.success


@pytest.mark.parametrize("task_id", CELLS)
def test_evaluating_the_untouched_workspace_earns_zero(task_id):
    _, obs = play(task_id, 11, [{"action_type": "exec", "command": "cat *"}, {"action_type": "evaluate"}])
    assert obs.done and obs.reward == 0.0


def test_a_partial_mutant_kill_earns_nothing():
    task = make_task("qe_mutant_kill-d3", 11)
    tests = kill_tests(task)
    assert len(tests) == 3
    _, obs = play("qe_mutant_kill-d3", 11, [
        {"action_type": "write_file", "file_path": "answer.json", "content": json.dumps({"tests": tests[:1]})},
        {"action_type": "evaluate"}])
    assert obs.done and obs.reward == 0.0


def test_a_wrong_expected_value_earns_nothing():
    task = make_task("qe_mutant_kill-d1", 11)
    tests = kill_tests(task, wrong_expected=True)
    _, obs = play("qe_mutant_kill-d1", 11, [
        {"action_type": "write_file", "file_path": "answer.json", "content": json.dumps({"tests": tests})},
        {"action_type": "evaluate"}])
    assert obs.reward == 0.0


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"tests": []}', "{" * 5000])
def test_malformed_answers_earn_zero(content):
    _, obs = play("qe_mutant_kill-d1", 11, [
        {"action_type": "write_file", "file_path": "answer.json", "content": content}, {"action_type": "evaluate"}])
    assert obs.done and obs.reward == 0.0


def test_a_fix_written_with_the_shell_counts_like_write_file():
    task = make_task("qe_bugfix-d2", 11)
    env = TerminalEnvironment()
    env.reset(seed=11, task_id="qe_bugfix-d2")
    env.step(act(action_type="exec", command=f"cat > {MODULE} <<'QE_EOF'\n{task.private['original']}QE_EOF"))
    assert env.step(act(action_type="evaluate")).reward == 1.0


def test_the_step_cap_grades_what_is_on_disk():
    task = make_task("qe_bugfix-d1", 11)
    env = TerminalEnvironment()
    env.reset(seed=11, task_id="qe_bugfix-d1")
    obs = env.step(act(action_type="write_file", file_path=MODULE, content=task.private["original"]))
    for _ in range(MAX_STEPS - 2):
        obs = env.step(act(action_type="exec", command="true"))
        assert not obs.done
    obs = env.step(act(action_type="exec", command="echo last"))
    assert obs.done and obs.reward == 1.0 and "last" in obs.output and "budget used up" in obs.output
    assert env.step(act(action_type="evaluate")).error.startswith("episode is terminal")


def _private_strings(task) -> list[str]:
    """Verifier material that must never appear in any observation field, decoded."""
    p = task.private
    found = [p["original"]] + [m["source"] for m in p.get("mutants", [])]
    found += [json.dumps(r.value) for r in p["reference"][:5] if r.kind == "ok" and len(json.dumps(r.value)) > 8]
    public = task.prompt + "".join(task.files.values())  # mutant-kill shows original.py; literals recur in code
    return [s for s in found if s not in public]


def _leaks(observations, secrets) -> list[str]:
    fields = [str(v) for obs in observations for v in obs.model_dump().values()]
    return [s for s in secrets if any(s in f for f in fields)]


def test_observations_never_carry_private_verifier_data():
    for task_id in ("qe_bugfix-d3", "qe_mutant_kill-d3"):
        task = make_task(task_id, 3)
        env = TerminalEnvironment()
        seen = [env.reset(seed=3, task_id=task_id)]
        seen.append(env.step(act(action_type="exec", command="ls; cat *")))
        seen.append(env.step(act(action_type="evaluate")))
        assert _leaks(seen, _private_strings(task)) == []


def test_the_leak_check_catches_an_injected_reference():  # positive control for the test above
    task = make_task("qe_bugfix-d3", 3)
    env = TerminalEnvironment()
    env.reset(seed=3, task_id="qe_bugfix-d3")
    env.step(act(action_type="write_file", file_path="ref.py", content=task.private["original"]))
    seen = [env.step(act(action_type="exec", command="cat ref.py"))]
    assert task.private["original"] in _leaks(seen, _private_strings(task))


def test_a_seeded_reset_replays_byte_for_byte():
    a = TerminalEnvironment().reset(seed=5, task_id="qe_bugfix-d2")
    b = TerminalEnvironment().reset(seed=5, task_id="qe_bugfix-d2")
    assert a.model_dump_json() == b.model_dump_json()
    assert "action_type" in a.instruction and "evaluate" in a.instruction


def test_the_shell_gets_a_minimal_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    env = TerminalEnvironment()
    env.reset(seed=1, task_id="qe_bugfix-d1")
    out = env.step(act(action_type="exec", command="env")).output
    assert "hf_secret_value" not in out and "HF_TOKEN" not in out and "PATH=/usr/local/bin:/usr/bin:/bin" in out


def test_exec_reports_exit_codes_truncates_output_and_times_out(monkeypatch):
    env = TerminalEnvironment()
    env.reset(seed=1, task_id="qe_bugfix-d1")
    obs = env.step(act(action_type="exec", command="echo hi; exit 3"))
    assert obs.output.startswith("hi") and not obs.success and obs.error == "exit code 3"
    obs = env.step(act(action_type="exec", command="head -c 100000 /dev/zero | tr '\\0' a"))
    assert obs.success and "[output truncated: 100000 bytes" in obs.output
    assert len(obs.output) < shell.MAX_OUTPUT_BYTES + 200
    monkeypatch.setattr(shell, "EXEC_TIMEOUT_S", 1.0)
    monkeypatch.setattr(shell.Workspace.exec, "__defaults__", (1.0,))
    obs = env.step(act(action_type="exec", command="sleep 20"))
    assert not obs.success and obs.error == "timed out" and "timed out after 1 s" in obs.output


def test_a_background_process_does_not_hold_the_command_open():
    env = TerminalEnvironment()
    env.reset(seed=1, task_id="qe_bugfix-d1")
    obs = env.step(act(action_type="exec", command="sleep 30 & echo started"))
    assert obs.success and "started" in obs.output and "timed out" not in obs.output


def test_the_workspace_is_removed_when_the_episode_ends():
    env = TerminalEnvironment()
    env.reset(seed=1, task_id="qe_bugfix-d1")
    path = env._workspace.path
    assert (path / MODULE).is_file()
    env.step(act(action_type="evaluate"))
    assert not path.exists()
    env.reset(seed=2, task_id="qe_bugfix-d1")
    second = env._workspace.path
    env.reset(seed=3, task_id="qe_bugfix-d1")
    assert not second.exists()
    env.close()


def test_an_infra_error_while_grading_leaves_the_episode_open(monkeypatch):
    task = make_task("qe_bugfix-d1", 11)
    env = TerminalEnvironment()
    env.reset(seed=11, task_id="qe_bugfix-d1")
    env.step(act(action_type="write_file", file_path=MODULE, content=task.private["original"]))
    real = terminal_env.grade_bugfix

    def flaky(*args):
        raise InfraError("worker died")

    monkeypatch.setattr(terminal_env, "grade_bugfix", flaky)
    with pytest.raises(InfraError):
        env.step(act(action_type="evaluate"))
    assert env.state.step_count == 1
    monkeypatch.setattr(terminal_env, "grade_bugfix", real)
    assert env.step(act(action_type="evaluate")).reward == 1.0


def test_disabled_mode_refuses_exec_but_write_and_evaluate_still_earn_one(disabled):
    task = make_task("qe_bugfix-d1", 11)
    env = TerminalEnvironment()
    obs = env.reset(seed=11, task_id="qe_bugfix-d1")
    assert "exec is unavailable" in obs.instruction
    obs = env.step(act(action_type="exec", command="touch pwned"))
    assert not obs.success and "unavailable" in obs.error and not (env._workspace.path / "pwned").exists()
    for bad in ("/etc/passwd", "../escape.py", "a/../../escape.py", ""):
        assert not env.step(act(action_type="write_file", file_path=bad, content="x")).success
    assert not (env._workspace.path.parent / "escape.py").exists()
    env.step(act(action_type="write_file", file_path=MODULE, content=task.private["original"]))
    assert env.step(act(action_type="evaluate")).reward == 1.0


def test_disabled_mode_never_follows_a_symlink_or_reads_a_fifo(disabled, tmp_path):
    secret = tmp_path / "reference.py"
    task = make_task("qe_bugfix-d1", 11)
    secret.write_text(task.private["original"])
    env = TerminalEnvironment()
    env.reset(seed=11, task_id="qe_bugfix-d1")
    module = env._workspace.path / MODULE
    module.unlink()
    module.symlink_to(secret)
    assert env._workspace.read(MODULE) is None
    module.unlink()
    os.mkfifo(module)
    assert env._workspace.read(MODULE) is None
    assert env.step(act(action_type="evaluate")).reward == 0.0


def test_a_write_to_a_fifo_nobody_reads_cannot_hang_the_step(monkeypatch):
    monkeypatch.setattr(shell, "IO_TIMEOUT_S", 2.0)
    env = TerminalEnvironment()
    env.reset(seed=1, task_id="qe_bugfix-d1")
    env.step(act(action_type="exec", command="mkfifo blocked"))
    started = __import__("time").monotonic()
    obs = env.step(act(action_type="write_file", file_path="blocked", content="x" * shell.MAX_WRITE_BYTES))
    assert __import__("time").monotonic() - started < 10 and not obs.success


def test_a_failed_build_removes_the_partial_workspace(monkeypatch, tmp_path):
    root = tmp_path / "ws"
    real = Path.write_text
    calls = []

    def fail_second(self, *a, **k):
        calls.append(self)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", fail_second)
    with pytest.raises(OSError):
        shell.Workspace({"a.py": "1", "b.py": "2"})
    assert list(root.iterdir()) == []
