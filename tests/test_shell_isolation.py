"""uid isolation of the terminal shell. Needs root on Linux, so it runs inside the terminal image:

    docker run --rm -v "$PWD/tests:/app/tests:ro" -v "$PWD/scripts:/app/scripts:ro" qe-arena-terminal:dev \
        sh -c 'pip install -q pytest==9.1.1 && python -m pytest -q -p no:cacheprovider /app/tests/test_shell_isolation.py'

Elsewhere every test is skipped.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux") or os.geteuid() != 0,
                                reason="uid isolation needs root on Linux (the terminal image)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from qe_arena import shell  # noqa: E402
from qe_arena.tasks import MODULE, make_task  # noqa: E402
from qe_arena.terminal_env import TerminalAction, TerminalEnvironment  # noqa: E402

PACKAGE = Path(shell.__file__).resolve().parent


@pytest.fixture(autouse=True)
def uid_mode(monkeypatch):
    monkeypatch.delenv(shell.ENV_FLAG, raising=False)
    shell._reset_mode_for_tests()
    assert shell.mode() == "uid"
    yield
    shell._reset_mode_for_tests()


def run(ws: shell.Workspace, command: str) -> shell.Result:
    return ws.exec(command)


def uid_processes(uid: int) -> list[int]:
    return shell._uid_pids(uid)


def test_commands_run_as_an_unprivileged_episode_uid():
    ws = shell.Workspace({"a.txt": "x"})
    try:
        out = run(ws, "id -u; id -G; stat -c %u .").output.split()
        assert int(out[0]) == ws.uid and ws.uid in shell.UID_POOL and out[1] == str(ws.uid) and out[2] == str(ws.uid)
    finally:
        ws.close()


def test_the_sources_and_graders_are_unreadable():
    ws = shell.Workspace({})
    try:
        for target in (PACKAGE / "tasks.py", PACKAGE / "corpus", Path(os.environ.get("QE_ARENA_WASI_ROOT", "/opt"))):
            result = run(ws, f"cat {target} || ls {target}")
            assert result.exit_code != 0 and "Permission denied" in result.output, target
        result = run(ws, "grep -rl def / --include=tasks.py 2>/dev/null | head")
        assert "qe_arena" not in result.output
    finally:
        ws.close()


def test_the_server_cannot_be_signalled_or_inspected():
    ws = shell.Workspace({})
    try:
        result = run(ws, f"kill -9 {os.getpid()}")
        assert result.exit_code != 0 and "not permitted" in result.output
        assert run(ws, f"cat /proc/{os.getpid()}/environ").exit_code != 0
    finally:
        ws.close()


def test_episodes_cannot_read_each_other():
    a, b = shell.Workspace({"secret.txt": "alpha"}), shell.Workspace({"x": ""})
    try:
        assert a.uid != b.uid
        result = run(b, f"cat {a.path}/secret.txt; ls {a.path}; ls {a.path.parent}")
        lines = result.output.splitlines()
        assert len(lines) == 3 and all("Permission denied" in line for line in lines), result.output
        assert "alpha" not in lines and "secret.txt" not in lines
    finally:
        a.close()
        b.close()


def test_background_processes_and_fork_bombs_are_cleaned_up():
    ws = shell.Workspace({})
    try:
        result = run(ws, "sleep 300 & setsid sleep 300 & echo ok")
        assert "ok" in result.output
        assert uid_processes(ws.uid) == []
        result = run(ws, "for i in $(seq 100); do sleep 5 & done; wait")
        assert "Resource temporarily unavailable" in result.output or "retry" in result.output
        assert uid_processes(ws.uid) == []
    finally:
        ws.close()


def test_memory_and_file_size_are_capped():
    ws = shell.Workspace({})
    try:
        result = run(ws, "python3 -c 'x = bytearray(2 * 1024**3)'")
        assert result.exit_code != 0 and "MemoryError" in result.output
        result = run(ws, "head -c 20000000 /dev/zero > big; ls -l big")
        assert result.exit_code != 0 or "16777216" in result.output
    finally:
        ws.close()


def test_a_planted_symlink_reads_only_what_the_episode_could():
    task = make_task("qe_bugfix-d1", 11)
    corpus_file = PACKAGE / "corpus" / f"{task.private['corpus']}.py"
    env = TerminalEnvironment()
    env.reset(seed=11, task_id="qe_bugfix-d1")
    env.step(TerminalAction(action_type="exec", command=f"ln -sf {corpus_file} {MODULE}"))
    assert env.step(TerminalAction(action_type="evaluate")).reward == 0.0


def test_writes_outside_the_workspace_fail():
    ws = shell.Workspace({})
    try:
        assert ws.write(str(PACKAGE / "pwned.py"), "x") is not None
        assert ws.write("/etc/pwned", "x") is not None
        assert not (PACKAGE / "pwned.py").exists() and not Path("/etc/pwned").exists()
        assert ws.write("sub/dir/ok.txt", "fine") is None and (ws.path / "sub/dir/ok.txt").read_text() == "fine"
    finally:
        ws.close()


def test_close_kills_a_detached_process_removes_everything_and_frees_the_uid():
    import subprocess

    ws = shell.Workspace({})
    uid, path = ws.uid, ws.path
    run(ws, "mkdir d && chmod 000 d .; true")  # before the sleeper: exec's own cleanup must not reach it
    # Started outside exec(), so only close() can remove it.
    sleeper = subprocess.Popen(["sleep", "300"], user=uid, group=uid, extra_groups=[], cwd="/",
                               start_new_session=True)
    time.sleep(0.2)
    assert sleeper.poll() is None, "the sleeper must be alive right before close()"
    ws.close()
    assert sleeper.wait(timeout=5) == -9  # close() killed it
    assert not path.exists() and uid_processes(uid) == [] and uid not in shell._UIDS_IN_USE
    assert uid not in shell._UIDS_QUARANTINED


def test_a_uid_whose_processes_survive_is_never_reused(monkeypatch):
    ws = shell.Workspace({})
    uid = ws.uid
    monkeypatch.setattr(shell, "_kill_uid", lambda u: False)
    ws.close()
    monkeypatch.undo()
    try:
        assert uid in shell._UIDS_QUARANTINED
        others = [shell.Workspace({}) for _ in range(3)]
        assert uid not in {o.uid for o in others}
        for o in others:
            o.close()
    finally:
        shell._UIDS_QUARANTINED.discard(uid)


def test_there_is_no_network_not_even_loopback_to_the_server():
    ws = shell.Workspace({})
    try:
        for target in ("('127.0.0.1', 8000)", "('1.1.1.1', 443)", "('::1', 8000)"):
            result = run(ws, f"python3 -c \"import socket; socket.create_connection({target}, timeout=3)\"")
            assert result.exit_code != 0 and "PermissionError" in result.output, result.output
        result = run(ws, "python3 -c \"import socket; socket.socketpair(socket.AF_INET)\"")  # domain-checked too
        assert result.exit_code != 0 and "PermissionError" in result.output, result.output
        result = run(ws, "python3 -c \"import socket; a, b = socket.socketpair(); a.send(b'x'); print(b.recv(1))\"")
        assert result.exit_code == 0 and "b'x'" in result.output  # AF_UNIX still works
        assert "netblock" not in run(ws, "git --version").output
    finally:
        ws.close()


def test_the_workspace_is_the_only_writable_place_and_nothing_survives_close():
    ws = shell.Workspace({})
    uid = ws.uid
    try:
        result = run(ws, "find / -xdev \\( -path /proc -o -path /sys \\) -prune -o -type d -writable -print 2>/dev/null; "
                         "for d in /tmp /var/tmp /run/lock /dev/shm /dev/mqueue; do touch $d/x 2>/dev/null && echo WROTE $d; done")
        writable = [line for line in result.output.splitlines() if line.strip()]
        assert all(line.startswith(str(ws.path)) for line in writable), writable
    finally:
        ws.close()
    for d in shell.SHARED_DIRS:
        if os.path.isdir(d):
            assert not any(e.stat(follow_symlinks=False).st_uid == uid for e in os.scandir(d))


def test_the_oracle_still_earns_one_under_isolation():
    task = make_task("qe_bugfix-d2", 11)
    env = TerminalEnvironment()
    env.reset(seed=11, task_id="qe_bugfix-d2")
    obs = env.step(TerminalAction(action_type="exec", command=f"python3 -c 'import module' && wc -l {MODULE}"))
    assert obs.success, obs.output
    env.step(TerminalAction(action_type="write_file", file_path=MODULE, content=task.private["original"]))
    assert env.step(TerminalAction(action_type="evaluate")).reward == 1.0


def test_system_v_ipc_is_denied():
    ws = shell.Workspace({})
    try:
        code = ("import ctypes; libc = ctypes.CDLL(None, use_errno=True); "
                "print([libc.shmget(0, 4096, 0o1666), libc.msgget(0, 0o1666), libc.semget(0, 1, 0o1666)], "
                "ctypes.get_errno())")
        result = run(ws, f'python3 -c "{code}"')
        assert result.exit_code == 0 and "[-1, -1, -1] 13" in result.output, result.output
    finally:
        ws.close()


def test_another_episode_cannot_read_a_running_command_from_proc():
    import threading

    a, b = shell.Workspace({}), shell.Workspace({})
    marker = "QE_SECRET_MARKER_9271"
    try:
        worker = threading.Thread(target=run, args=(a, f": {marker}; sleep 3"))
        worker.start()
        time.sleep(1.0)
        assert uid_processes(a.uid), "the command must be running while b looks"
        seen = run(b, "cat /proc/[0-9]*/cmdline /proc/[0-9]*/environ 2>/dev/null | tr '\\0' ' '")
        worker.join()
        assert marker not in seen.output
    finally:
        a.close()
        b.close()


def _live_tasks(uid: int) -> list[str]:
    """Every live thread of ``uid``, read independently of shell._uid_pids."""
    live = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != uid:
                continue
            for task in (proc / "task").iterdir():
                state = (task / "stat").read_bytes().rsplit(b")", 1)[1].split()[0]
                if state not in (b"Z", b"X"):
                    live.append(f"{proc.name}/{task.name}")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return live


def test_a_zombie_leader_with_a_live_thread_is_still_killed():
    ws = shell.Workspace({})
    code = ("import threading, time, ctypes; threading.Thread(target=time.sleep, args=(300,)).start(); "
            "time.sleep(0.2); ctypes.CDLL(None).pthread_exit(None)")
    try:
        run(ws, f'(python3 -c "{code}" &) ; sleep 1; echo started')
        assert _live_tasks(ws.uid) == []
    finally:
        ws.close()


def test_a_process_name_that_is_not_utf8_does_not_break_cleanup():
    ws = shell.Workspace({})
    code = "import ctypes, time; ctypes.CDLL(None).prctl(15, bytes([255, 254, 41, 41]), 0, 0, 0); time.sleep(300)"
    try:
        result = run(ws, f'(python3 -c "{code}" &) ; sleep 0.5; cat /proc/[0-9]*/comm | od -c | head -3')
        assert "377" in result.output, "the odd name must really have been set"
        assert _live_tasks(ws.uid) == []
    finally:
        ws.close()
    assert ws.uid not in shell._UIDS_QUARANTINED



def test_cleanup_does_not_trust_a_scan_that_missed_a_process(monkeypatch):
    """A /proc scan can miss a child forked after the directory was read (the parent then exits); the kill
    must reach every process of the uid anyway, not only the pids a scan returned."""
    import subprocess

    ws = shell.Workspace({})
    uid = ws.uid
    sleeper = subprocess.Popen(["sleep", "300"], user=uid, group=uid, extra_groups=[], cwd="/",
                               start_new_session=True)
    time.sleep(0.2)
    real_scan, calls = shell._uid_pids, []

    def scan_that_misses_once(u):
        calls.append(u)
        return [] if len(calls) == 1 else real_scan(u)

    monkeypatch.setattr(shell, "_uid_pids", scan_that_misses_once)
    try:
        assert shell._kill_uid(uid)
        assert sleeper.wait(timeout=5) == -9
    finally:
        monkeypatch.undo()
        if sleeper.poll() is None:
            sleeper.kill()
        ws.close()


def test_an_empty_scan_never_counts_when_the_kill_helper_failed(monkeypatch):
    # e.g. the episode killed or stopped the helper before its kill(-1)
    monkeypatch.setattr(shell, "_kill_as", lambda uid: False)
    monkeypatch.setattr(shell, "_uid_pids", lambda uid: [])
    assert shell._kill_uid(shell.UID_POOL[0]) is False



def test_the_episode_cannot_signal_trace_or_read_the_kill_helper():
    import subprocess

    ws = shell.Workspace({})
    killer = ws.uid + shell.KILLER_UID_OFFSET
    helper_like = subprocess.Popen(  # the helper's credentials, kept alive long enough to probe
        [sys.executable, "-I", "-S", "-c",
         f"import os, time; os.setgroups([]); os.setresgid({killer}, {killer}, {killer}); "
         f"os.setresuid({killer}, {ws.uid}, {killer}); time.sleep(30)"], env={"QE_SECRET": "s3cret"})
    time.sleep(0.3)
    pid = helper_like.pid
    try:
        result = run(ws, f"kill -STOP {pid}; kill -9 {pid}; cat /proc/{pid}/environ; "
                         f"grep -q '^State:.*[RS]' /proc/{pid}/status && echo STILL_RUNNING; "
                         "python3 -c \"import ctypes; print('attach', ctypes.CDLL(None).ptrace(16, " + str(pid)
                         + ", 0, 0))\"")
        assert result.output.count("not permitted") >= 2, result.output
        assert "s3cret" not in result.output and "attach -1" in result.output, result.output
        assert "STILL_RUNNING" in result.output, result.output  # neither stopped nor killed by the episode
    finally:
        helper_like.kill()
        helper_like.wait()
        ws.close()


def test_a_kill_helper_that_hangs_or_cannot_start_fails_closed(monkeypatch):
    monkeypatch.setattr(shell, "KILL_HELPER_TIMEOUT_S", 0.3)
    monkeypatch.setattr(shell, "_KILL_ALL", "import time; time.sleep(30)")
    started = time.monotonic()
    assert shell._kill_as(shell.UID_POOL[0]) is False and time.monotonic() - started < 3
    monkeypatch.undo()

    def no_fork(*args, **kwargs):
        raise BlockingIOError(11, "Resource temporarily unavailable")

    monkeypatch.setattr(shell.subprocess, "Popen", no_fork)
    assert shell._kill_uid(shell.UID_POOL[0]) is False
    monkeypatch.undo()

    def scan_error(uid):
        raise OSError("proc unreadable")

    monkeypatch.setattr(shell, "_uid_pids", scan_error)
    assert shell._kill_uid(shell.UID_POOL[0]) is False

def test_a_failed_per_command_cleanup_quarantines_the_uid_at_close(monkeypatch):
    ws = shell.Workspace({})
    uid = ws.uid
    try:
        monkeypatch.setattr(shell, "_kill_uid", lambda u: False)
        run(ws, "true")
        monkeypatch.undo()
        ws.close()
        assert uid in shell._UIDS_QUARANTINED
    finally:
        monkeypatch.undo()
        ws.close()
        shell._UIDS_QUARANTINED.discard(uid)
