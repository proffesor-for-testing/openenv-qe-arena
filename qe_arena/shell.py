"""A real bash shell per episode, for exploring only: the reward never depends on anything it prints.

Isolation comes from the operating system, not from parsing commands:

* ``uid`` mode (the terminal image, whose server runs as root): every episode gets its own unprivileged uid
  from ``UID_POOL`` and a 0700 workspace owned by it. Commands, file writes and the read of the graded file
  all run as that uid, so it cannot read the image's sources (0700 root), other episodes' workspaces or the
  server's memory and environment, and cannot signal the server. The shared world-writable directories are
  locked (``SHARED_DIRS``), so the workspace is the only place an episode can write, and anything the uid
  still owns elsewhere is removed when the episode closes. A seccomp filter (``NETBLOCK``) allows only
  AF_UNIX sockets, so there is no network, loopback to the server included. Each command runs under ulimits
  with a minimal environment (no inherited secrets) and umask 077, and every process the uid left behind is
  killed afterwards. A uid whose processes could not be killed is never reused.
* ``unsandboxed`` mode (tests and local development only, opt-in with ``QE_ARENA_UNSANDBOXED_SHELL=1``):
  the same code path as the server's own user, with no isolation. Never set this on a shared host.
* ``disabled`` mode (anything else, e.g. a node that drops CAP_SETUID, board #213, or one where any part
  of the uid-mode probe fails): ``exec`` is refused, while ``write_file`` and the graded-file read happen
  in-process on validated relative paths. Without ``exec`` nothing the model controls can plant a symlink or
  FIFO, so that is safe, and the episode still trains the write-then-evaluate habit instead of scoring 0.

Not isolated, for want of privileges (no cgroups or namespaces): total disk use and inode count inside the
workspace (only per file, ``LIMITS``), total memory across an episode's processes (only per process), the
argv of programs a command starts (``/proc/<pid>/cmdline`` is world-readable without hidepid), and abstract
AF_UNIX sockets. The last two let concurrently running episodes cooperate if both try; neither reaches the
grader, the hidden tests or the reference sources.
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

EXEC_TIMEOUT_S = 30.0
IO_TIMEOUT_S = 10.0
EXIT_GRACE_S = 0.5  # after the command exits, how long background holders of its output may keep writing
MAX_OUTPUT_BYTES = 8192  # about 2-3k tokens: a careless cat still costs budget, as in the eval
MAX_COMMAND_BYTES = 16384
MAX_WRITE_BYTES = 65536
MAX_READ_BYTES = 65536
MAX_PATH_CHARS = 256
UID_POOL = range(20000, 20064)
# Per command: CPU seconds, virtual memory (KiB), largest file (KiB), processes of the episode uid.
LIMITS = (("t", 30), ("v", 786432), ("f", 16384), ("u", 32))
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"
ENV_FLAG = "QE_ARENA_UNSANDBOXED_SHELL"
ROOT_ENV = "QE_ARENA_WORKSPACES"
# World-writable directories of the image and its runtime mounts. Dockerfile.terminal locks the image's own;
# the probe locks the mounts (tmpfs /tmp, /dev/shm, /dev/mqueue) and checks that an episode uid can write none.
SHARED_DIRS = ("/tmp", "/var/tmp", "/run/lock", "/dev/shm", "/dev/mqueue")

# Runs as the episode uid, before bash: no_new_privs, then a seccomp filter, then exec of the rest of argv.
# Denied with EACCES: socket() and socketpair() for any family but AF_UNIX (so no network, loopback to the
# server included), io_uring (it can open sockets), System V IPC and POSIX message queues (objects that outlive
# the episode). Killed: another architecture, and syscall numbers >= 512 (x32 and the old x32 range). Jumps are
# assembled from labels. Unprivileged by design; if it cannot be installed the probe selects disabled mode.
NETBLOCK = r"""
import ctypes, os, platform, struct, sys
NRS = {"x86_64": (0xC000003E, 41, 53, (425, 29, 30, 31, 64, 65, 66, 68, 69, 70, 71, 220, 240)),
       "aarch64": (0xC00000B7, 198, 199, (425, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 180))}
spec = NRS.get(platform.machine())
if spec is None:
    sys.stderr.write("netblock: unsupported architecture\n"); os._exit(126)
audit, socket_nr, socketpair_nr, denied = spec
LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
code = [("ld", 4), ("jeq", audit, None, "kill"), ("ld", 0), ("jge", 512, "kill", None)]
code += [("jeq", nr, "deny", None) for nr in denied]
code += [("jeq", socket_nr, "family", None), ("jeq", socketpair_nr, "family", "allow")]
code += [("label", "family"), ("ld", 16), ("jeq", 1, "allow", "deny")]
code += [("label", "allow"), ("ret", 0x7FFF0000), ("label", "deny"), ("ret", 0x00050000 | 13),
         ("label", "kill"), ("ret", 0x80000000)]
labels, n = {}, 0
for op in code:
    if op[0] == "label":
        labels[op[1]] = n
    else:
        n += 1
out, i = [], 0
for op in code:
    if op[0] == "label":
        continue
    rel = lambda t: 0 if t is None else labels[t] - i - 1
    if op[0] == "ld":
        out.append((LD, 0, 0, op[1]))
    elif op[0] == "ret":
        out.append((RET, 0, 0, op[1]))
    else:
        out.append((JEQ if op[0] == "jeq" else JGE, rel(op[2]), rel(op[3]), op[1]))
    i += 1
buf = ctypes.create_string_buffer(b"".join(struct.pack("HBBI", *ins) for ins in out))
class Prog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
libc = ctypes.CDLL(None, use_errno=True)
fprog = Prog(len(out), ctypes.addressof(buf))
if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(fprog), 0, 0) != 0:
    sys.stderr.write("netblock: seccomp unavailable: errno %d\n" % ctypes.get_errno()); os._exit(126)
os.execv(sys.argv[1], sys.argv[1:])
"""
_MODE: list[str] = []
_MODE_LOCK = threading.Lock()
_UIDS_LOCK = threading.Lock()
_UIDS_IN_USE: set[int] = set()
_UIDS_QUARANTINED: set[int] = set()


def mode() -> str:
    """``uid``, ``unsandboxed`` or ``disabled``; probed once per process."""
    with _MODE_LOCK:
        if not _MODE:
            _MODE.append(_probe())
            log.warning("shell isolation mode: %s", _MODE[0])
        return _MODE[0]


def _netblocked(argv: list[str]) -> list[str]:
    return [sys.executable, "-I", "-S", "-c", NETBLOCK, *argv]


def _lock_shared_dirs() -> None:
    for name in SHARED_DIRS:
        try:
            st = os.lstat(name)
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(st.st_mode) and st.st_mode & 0o002:
            os.chmod(name, 0o711)  # traversable, so the workspace root below /tmp stays reachable


def _probe() -> str:
    if os.environ.get(ENV_FLAG) == "1":
        return "unsandboxed"
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0 or not Path("/proc").is_dir():
        return "disabled"
    uid = UID_POOL[-1]
    try:
        _lock_shared_dirs()
        probe_dir = Path(tempfile.mkdtemp(prefix="probe-", dir=_workspace_root()))
        try:
            os.chown(probe_dir, uid, uid)  # CAP_CHOWN
            script = ('test "$(id -u)" = ' + str(uid) + ' || exit 3; '
                      + "".join(f"touch {d}/.qe-probe 2>/dev/null && exit 4; " for d in SHARED_DIRS)
                      + "python3 -c 'import socket; socket.socket()' 2>/dev/null && exit 5; "
                      + "setsid sleep 60 >/dev/null 2>&1 & echo $! > pid; touch ok")
            probe = subprocess.run(_netblocked(["/bin/sh", "-c", script]), cwd=probe_dir, user=uid, group=uid,
                                   extra_groups=[], env={"PATH": SAFE_PATH}, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=IO_TIMEOUT_S,
                                   check=False)
            if probe.returncode != 0 or not (probe_dir / "ok").is_file():
                log.warning("shell probe failed: exit %s %s", probe.returncode, probe.stderr[-200:])
                return "disabled"
            pid = int((probe_dir / "pid").read_text())
            if not Path(f"/proc/{pid}").exists() or os.stat(f"/proc/{pid}").st_uid != uid:
                log.warning("shell probe failed: the detached process did not start")
                return "disabled"
            killed = _kill_uid(uid)  # CAP_KILL: it must be killable; tini then reaps it
            for _ in range(50):
                if not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(0.05)
            if not killed or _uid_pids(uid):
                log.warning("shell probe failed: a detached process survived SIGKILL")
                return "disabled"
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        log.warning("shell probe failed: %s", exc)
        _kill_uid(uid)
        return "disabled"
    return "uid"


def _reset_mode_for_tests() -> None:
    with _MODE_LOCK:
        _MODE.clear()


@dataclass
class Result:
    output: str
    exit_code: int | None  # None: timed out or could not start
    timed_out: bool = False


def _workspace_root() -> Path:
    root = Path(os.environ.get(ROOT_ENV) or Path(tempfile.gettempdir()) / "qe-arena-ws")
    root.mkdir(mode=0o711, parents=True, exist_ok=True)
    return root


def _acquire_uid() -> int:
    with _UIDS_LOCK:
        for uid in UID_POOL[:-1]:  # the last one is the probe's
            if uid not in _UIDS_IN_USE and uid not in _UIDS_QUARANTINED:
                _UIDS_IN_USE.add(uid)
                return uid
    raise RuntimeError("no free shell uid: too many live episodes")


def _release_uid(uid: int, clean: bool) -> None:
    with _UIDS_LOCK:
        _UIDS_IN_USE.discard(uid)
        if not clean:
            _UIDS_QUARANTINED.add(uid)
            log.error("shell uid %s quarantined: its processes or files could not be removed", uid)


def _state(path: Path) -> bytes | None:
    """The state letter from a /proc stat file (bytes: a process name need not be UTF-8), or None if gone."""
    try:
        fields = path.read_bytes().rsplit(b")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None
    return fields[0] if fields else None


def _uid_pids(uid: int) -> list[int]:
    """Processes of ``uid`` with any live thread. A zombie thread-group leader whose other threads still run
    counts as live (signalling its pid kills the whole group); a fully dead zombie does not, since only its
    parent can reap it."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
        except FileNotFoundError:
            continue
        state = _state(entry / "stat")
        if state is None:
            continue
        if state in (b"Z", b"X"):
            try:
                tasks = list((entry / "task").iterdir())
            except (FileNotFoundError, ProcessLookupError):
                continue
            if not any(_state(t / "stat") not in (None, b"Z", b"X") for t in tasks):
                continue
        pids.append(int(entry.name))
    return pids


def _kill_uid(uid: int) -> bool:
    """SIGKILL every process owned by ``uid`` (Linux /proc); True once none is left."""
    for _ in range(20):
        pids = _uid_pids(uid)
        if not pids:
            return True
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError:
                return False
        time.sleep(0.05)
    return not _uid_pids(uid)


def _sweep_uid_files(uid: int) -> bool:
    """Remove what ``uid`` owns in the shared directories; normally nothing, since they are locked."""
    clean = True
    for name in SHARED_DIRS:
        try:
            entries = list(os.scandir(name))
        except FileNotFoundError:
            continue
        except OSError:
            clean = False
            continue
        for entry in entries:
            try:
                if entry.stat(follow_symlinks=False).st_uid != uid:
                    continue
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.unlink(entry.path)
            except FileNotFoundError:
                continue
            except OSError:
                clean = False
    return clean


def _valid_relative(path: str) -> str | None:
    """A normalised relative path inside the workspace, or None."""
    if not path or len(path) > MAX_PATH_CHARS or "\0" in path:
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if path.startswith("/") or not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def _feed(pipe, data: bytes) -> None:
    """Write stdin from a thread, so a reader that never reads cannot block the caller past its deadline."""
    try:
        pipe.write(data)
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except (BrokenPipeError, OSError):
            pass


class Workspace:
    """One episode's directory and, in ``uid`` mode, its uid. ``close`` removes both."""

    def __init__(self, files: dict[str, str]):
        self._closed = True  # until fully built: close() then has nothing to undo
        self.path: Path | None = None
        self.mode = mode()
        self.uid = _acquire_uid() if self.mode == "uid" else None
        try:
            self.path = Path(tempfile.mkdtemp(prefix="ep-", dir=_workspace_root()))
            for name, text in files.items():
                target = self.path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            if self.uid is not None:
                for dirpath, dirnames, filenames in os.walk(self.path):
                    for name in dirnames + filenames:
                        os.chown(os.path.join(dirpath, name), self.uid, self.uid)
                os.chown(self.path, self.uid, self.uid)
            os.chmod(self.path, 0o700)
        except BaseException:
            removed = True
            if self.path is not None:
                shutil.rmtree(self.path, ignore_errors=True)
                removed = not self.path.exists()
            if self.uid is not None:
                _release_uid(self.uid, clean=removed)  # nothing ran as it, but files may already be its
            raise
        self._closed = False

    @property
    def exec_available(self) -> bool:
        return self.mode in ("uid", "unsandboxed")

    def _env(self) -> dict[str, str]:
        home = str(self.path)
        return {"PATH": SAFE_PATH, "HOME": home, "TMPDIR": home, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "TERM": "dumb", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}

    def _spawn(self, argv: list[str], stdin: bool, extra_env: dict[str, str] | None = None) -> subprocess.Popen:
        kwargs = {}
        if self.uid is not None:
            kwargs = {"user": self.uid, "group": self.uid, "extra_groups": [], "umask": 0o077}
            argv = _netblocked(argv)
        return subprocess.Popen(argv, cwd=self.path, env={**self._env(), **(extra_env or {})},
                                start_new_session=True, close_fds=True,
                                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)

    def _run(self, argv: list[str], timeout_s: float, cap: int, stdin: bytes | None = None,
             extra_env: dict[str, str] | None = None) -> tuple[bytes, int, int | None, bool]:
        """(first ``cap`` bytes of output, total bytes, exit code, timed out)."""
        deadline = time.monotonic() + timeout_s
        proc = self._spawn(argv, stdin is not None, extra_env)
        kept, total, timed_out, exited_at = bytearray(), 0, False, None
        try:
            if stdin is not None:
                threading.Thread(target=_feed, args=(proc.stdin, stdin), daemon=True).start()
            fd = proc.stdout.fileno()
            while True:
                now = time.monotonic()
                if exited_at is None and proc.poll() is not None:
                    exited_at = now
                if now >= deadline:
                    timed_out = proc.poll() is None
                    break
                if exited_at is not None and now - exited_at >= EXIT_GRACE_S:
                    break
                limit = deadline if exited_at is None else min(deadline, exited_at + EXIT_GRACE_S)
                ready, _, _ = select.select([fd], [], [], max(0.0, min(limit - now, 0.1)))
                if not ready:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    if exited_at is None:
                        try:
                            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                        except subprocess.TimeoutExpired:
                            timed_out = True
                    break
                total += len(chunk)
                if len(kept) < cap:
                    kept.extend(chunk[:cap - len(kept)])
        finally:
            self._kill(proc)  # also unblocks the stdin thread: its reader is gone
            proc.stdout.close()
        code = None if timed_out else proc.returncode
        return bytes(kept), total, code, timed_out

    def _kill(self, proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if self.uid is not None and not _kill_uid(self.uid):  # background processes that left the group, too
            log.error("shell uid %s has processes that SIGKILL did not remove", self.uid)
        try:
            proc.wait(timeout=IO_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.error("shell process %s did not exit after SIGKILL", proc.pid)

    def exec(self, command: str, timeout_s: float = EXEC_TIMEOUT_S) -> Result:
        if not self.exec_available:
            return Result("exec is unavailable on this host; use write_file and evaluate", None)
        if not command.strip():
            return Result("empty command", None)
        if len(command.encode("utf-8", "surrogatepass")) > MAX_COMMAND_BYTES or "\0" in command:
            return Result(f"command too long (limit {MAX_COMMAND_BYTES} bytes)", None)
        # RLIMIT_NPROC counts every process of the real uid, so it is only meaningful for an episode's own uid.
        limits = "; ".join(f"ulimit -{flag} {value} 2>/dev/null" for flag, value in LIMITS
                           if flag != "u" or self.uid is not None)
        # The command travels in the environment, not argv: /proc/<pid>/environ is private to the uid, while
        # cmdline is world-readable. (What the command itself starts, e.g. python3 -c, is in argv as usual.)
        wrapper = (f"{limits}; {{ echo 1000 > /proc/self/oom_score_adj; }} 2>/dev/null; "
                   'qe_command=$QE_COMMAND; unset QE_COMMAND; eval "$qe_command"')
        out, total, code, timed_out = self._run(["/bin/bash", "--noprofile", "--norc", "-c", wrapper, "qe-shell"],
                                                timeout_s, MAX_OUTPUT_BYTES, extra_env={"QE_COMMAND": command})
        text = out.decode("utf-8", "replace")
        if total > len(out):
            text += f"\n[output truncated: {total} bytes, first {len(out)} shown]"
        if timed_out:
            text += f"\n[command timed out after {timeout_s:g} s and was killed]"
        return Result(text, code, timed_out)

    def write(self, file_path: str, content: str) -> str | None:
        """Write a whole file; an error message, or None on success."""
        data = content.encode("utf-8", "replace")
        if len(data) > MAX_WRITE_BYTES:
            return f"content too large (limit {MAX_WRITE_BYTES} bytes)"
        if not file_path or len(file_path) > MAX_PATH_CHARS or "\0" in file_path:
            return "file_path must be a non-empty path"
        if not self.exec_available:
            rel = _valid_relative(file_path)
            if rel is None:
                return "file_path must be a relative path inside the working directory"
            return self._write_in_process(rel, data)
        script = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'
        out, _, code, timed_out = self._run(["/bin/sh", "-c", script, "qe-write", file_path], IO_TIMEOUT_S, 2048,
                                            stdin=data)
        if timed_out or code != 0:
            return "could not write file: " + (out.decode("utf-8", "replace").strip() or "timed out")[:300]
        return None

    def _write_in_process(self, rel: str, data: bytes) -> str | None:
        target = self.path / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        except OSError as exc:
            return f"could not write file: {exc.strerror}"
        return None

    def read(self, rel: str) -> str | None:
        """The graded file's text, read with the episode's own permissions; None if missing or unreadable."""
        if _valid_relative(rel) != rel:
            raise ValueError("graded paths are fixed relative names")
        if not self.exec_available:
            try:
                fd = os.open(self.path / rel, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            except OSError:
                return None
            with os.fdopen(fd, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    return None
                data = handle.read(MAX_READ_BYTES + 1)
        else:
            # As the episode uid, so a symlink or FIFO the model planted reads only what that uid could.
            script = 'test -f "$1" && exec cat -- "$1"'
            data, total, code, timed_out = self._run(["/bin/sh", "-c", script, "qe-read", rel], IO_TIMEOUT_S,
                                                     MAX_READ_BYTES + 1)
            if timed_out or code != 0:
                return None
        if len(data) > MAX_READ_BYTES:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        clean = True
        if self.uid is not None:
            try:
                clean = _kill_uid(self.uid) and _sweep_uid_files(self.uid)
            except Exception:  # noqa: BLE001 - an unexpected scan failure must still free the directory
                log.exception("shell uid %s cleanup failed", self.uid)
                clean = False
        shutil.rmtree(self.path, ignore_errors=True)
        clean = clean and not self.path.exists()
        if self.uid is not None:
            _release_uid(self.uid, clean)

    def __del__(self):  # a dropped environment must not leak its uid or directory
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
