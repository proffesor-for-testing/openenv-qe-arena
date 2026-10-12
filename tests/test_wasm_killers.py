"""Mutation killers for the WASI sandbox, from the verifier-mutation run on b43f503 (64.6% -> 97.8%).

Adapted to the current API (deadline-bound _acquire, cgroup-sized pool, skipped kind, InfraError),
plus killers for the HIGH gaps the run reported: the partial-fix guard, fuel/resource never killing,
and warm()/pool slot accounting.
"""

from __future__ import annotations

import struct

import pytest

from qe_arena import wasi_fs
from qe_arena.wasi_fs import VirtualFS

ZIP = "/lib/python314.zip"


class _Memory:
    def __init__(self, size=8192):
        self.data = bytearray(size)

    def read(self, caller, start, stop):
        if not 0 <= start <= stop <= len(self.data):
            raise IndexError("out of bounds")
        return self.data[start:stop]

    def write(self, caller, data, ptr):
        if not 0 <= ptr <= ptr + len(data) <= len(self.data):
            raise IndexError("out of bounds")
        self.data[ptr:ptr + len(data)] = data


class _Caller:
    def __init__(self):
        self.memory = _Memory()

    def get(self, name):
        assert name == "memory"
        return self.memory


def _fs(files=None):
    writes = []
    fs = VirtualFS(files or {ZIP: b"0123456789"}, lambda c, fd, data: writes.append((fd, data)) or 0)
    return fs, _Caller(), writes


def _put(caller, ptr, data):
    caller.memory.write(caller, data, ptr)
    return len(data)


def _u32(caller, ptr):
    return int.from_bytes(caller.memory.data[ptr:ptr + 4], "little")


def _open(fs, caller, path, dirfd=3, oflags=0, rights=0, fdflags=0):
    n = _put(caller, 100, path.encode())
    errno = fs.path_open(caller, dirfd, 0, 100, n, oflags, rights, 0, fdflags, 200)
    return errno, _u32(caller, 200)


# ---------------------------------------------------------------- ABI constants

def test_wasi_preview1_constants_are_the_abi_values():
    # Kills the errno/filetype/oflag/fdflag/rights/whence const_off_by_one survivors: the tests used the
    # module's own names, so a wrong number was invisible unless the interpreter happened to branch on it.
    assert (wasi_fs.SUCCESS, wasi_fs.EACCES, wasi_fs.EBADF, wasi_fs.EINVAL, wasi_fs.EISDIR, wasi_fs.ENOENT,
            wasi_fs.ENOTDIR, wasi_fs.ENOTSUP, wasi_fs.EPERM, wasi_fs.EROFS, wasi_fs.ESPIPE) == (
        0, 2, 8, 28, 31, 44, 54, 58, 63, 69, 70)
    assert (wasi_fs.DIRECTORY, wasi_fs.REGULAR_FILE) == (3, 4)
    assert (wasi_fs.O_CREAT, wasi_fs.O_DIRECTORY, wasi_fs.O_TRUNC) == (1, 2, 8)
    assert wasi_fs.FDFLAG_APPEND == 1 and wasi_fs.RIGHT_FD_WRITE == 64
    assert (wasi_fs.WHENCE_SET, wasi_fs.WHENCE_CUR, wasi_fs.WHENCE_END) == (0, 1, 2)
    # fd_read(1), fd_seek(2), fd_tell(5), fd_advise(7), fd_filestat_get(21): read-only file rights
    assert wasi_fs.RIGHTS_FILE == 2 + 4 + 32 + 128 + 2097152
    assert wasi_fs.RIGHTS_DIR == 0x1FFFFFFF & ~64 and not wasi_fs.RIGHTS_DIR & 64
    assert wasi_fs.FIXED_TIME_NS == 1_700_000_000_000_000_000


# ---------------------------------------------------------------- path resolution

@pytest.mark.parametrize("path, errno", [
    ("/lib/python314.zip", wasi_fs.EPERM),     # absolute, even inside the preopen
    ("../libx", wasi_fs.EPERM),                # sibling with a shared prefix
    ("../lib/python314.zip", 0),               # normalizes back inside
    ("./python314.zip", 0),
    ("", 0),                                   # the preopen itself
])
def test_resolve_refuses_absolute_and_prefix_sibling_paths(path, errno):
    # Kills M-fs-resolve-allow-abs, M-fs-resolve-prefix-loose.
    fs, caller, _ = _fs()
    assert _open(fs, caller, path)[0] == errno


def test_resolve_needs_an_open_directory_fd():
    fs, caller, _ = _fs()
    _, fd = _open(fs, caller, "python314.zip")
    assert _open(fs, caller, "x", dirfd=fd)[0] == wasi_fs.ENOTDIR
    assert _open(fs, caller, "x", dirfd=9)[0] == wasi_fs.EBADF


def test_nested_files_create_their_directories_and_list_correctly():
    # Kills E-wasi_fs-58 (dir creation loop), E-wasi_fs-217 (readdir filter) and E-wasi_fs-220 (dtype).
    fs, caller, _ = _fs({"/lib/a/b.py": b"x", ZIP: b"z"})
    assert fs.dirs == {"/lib", "/lib/a"}
    errno, fd = _open(fs, caller, "a", oflags=wasi_fs.O_DIRECTORY)
    assert errno == 0
    assert fs.fd_readdir(caller, 3, 1000, 4000, 0, 1500) == 0
    entries = _readdir_entries(caller, 1000, _u32(caller, 1500))
    assert [(n, t) for n, _, _, t in entries] == [
        (".", wasi_fs.DIRECTORY), ("..", wasi_fs.DIRECTORY), ("a", wasi_fs.DIRECTORY),
        ("python314.zip", wasi_fs.REGULAR_FILE)]


def _readdir_entries(caller, ptr, used):
    raw, out, off = bytes(caller.memory.data[ptr:ptr + used]), [], 0
    while off + 24 <= len(raw):
        nxt, ino, namlen, dtype = struct.unpack_from("<QQIB", raw, off)
        out.append((raw[off + 24:off + 24 + namlen].decode(), nxt, ino, dtype))
        off += 24 + namlen
    return out


def test_readdir_inodes_types_and_negative_cookie():
    # Kills E-wasi_fs-225 (".." default inode), E-wasi_fs-220, E-wasi_fs-222, M-fs-readdir-cookie-ignored.
    fs, caller, _ = _fs()
    assert fs.fd_readdir(caller, 3, 1000, 4000, 0, 1500) == 0
    entries = _readdir_entries(caller, 1000, _u32(caller, 1500))
    assert [(n, c, i, t) for n, c, i, t in entries] == [
        (".", 1, fs.inodes["/lib"], wasi_fs.DIRECTORY), ("..", 2, 1, wasi_fs.DIRECTORY),
        ("python314.zip", 3, fs.inodes[ZIP], wasi_fs.REGULAR_FILE)]
    assert fs.fd_readdir(caller, 3, 1000, 4000, -5, 1500) == 0  # a negative cookie restarts
    assert _readdir_entries(caller, 1000, _u32(caller, 1500))[0][0] == "."
    assert fs.fd_readdir(caller, 3, 1000, 4000, 3, 1500) == 0 and _u32(caller, 1500) == 0  # past the end
    _, fd = _open(fs, caller, "python314.zip")
    assert fs.fd_readdir(caller, fd, 1000, 4000, 0, 1500) == wasi_fs.ENOTDIR
    assert fs.fd_readdir(caller, 9, 1000, 4000, 0, 1500) == wasi_fs.EBADF


# ---------------------------------------------------------------- fds

def test_two_files_open_at_once_get_distinct_fresh_fds():
    # Kills E-wasi_fs-267/268/269 (fd allocation: start, loop condition, step).
    fs, caller, _ = _fs()
    assert _open(fs, caller, "python314.zip") == (0, 4)
    assert _open(fs, caller, "python314.zip") == (0, 5)
    assert fs.fd_close(caller, 4) == 0
    assert _open(fs, caller, "python314.zip") == (0, 4)


def test_multi_iovec_reads_fill_each_buffer_in_turn():
    # Kills E-wasi_fs-77/78 (iovec stride) and E-wasi_fs-141 (short-read boundary).
    fs, caller, _ = _fs()
    _, fd = _open(fs, caller, "python314.zip")
    iovs = struct.pack("<IIII", 300, 4, 320, 3)
    _put(caller, 400, iovs)
    assert fs.fd_read(caller, fd, 400, 2, 500) == 0
    assert bytes(caller.memory.data[300:304]) == b"0123" and bytes(caller.memory.data[320:323]) == b"456"
    assert _u32(caller, 500) == 7
    assert fs.fd_seek(caller, fd, 2, wasi_fs.WHENCE_CUR, 600) == 0  # literal ABI whence 1 is CUR
    assert int.from_bytes(caller.memory.data[600:608], "little") == 9
    assert fs.fd_seek(caller, fd, 0, 1, 600) == 0 and int.from_bytes(caller.memory.data[600:608], "little") == 9
    assert fs.fd_seek(caller, fd, 0, 7, 600) == wasi_fs.EINVAL


def test_pread_does_not_move_the_position_and_refuses_stdio_and_dirs():
    # Kills E-wasi_fs-119 (stdio set) and M-fs-pread-moves-pos.
    fs, caller, _ = _fs()
    _, fd = _open(fs, caller, "python314.zip")
    _put(caller, 400, struct.pack("<II", 300, 4))
    assert fs.fd_pread(caller, fd, 400, 1, 5, 500) == 0 and bytes(caller.memory.data[300:304]) == b"5678"
    assert fs.fd_tell(caller, fd, 600) == 0 and int.from_bytes(caller.memory.data[600:608], "little") == 0
    for std in (0, 1, 2):
        assert fs.fd_pread(caller, std, 400, 1, 0, 500) == wasi_fs.ESPIPE
    assert fs.fd_pread(caller, 3, 400, 1, 0, 500) == wasi_fs.EISDIR
    assert fs.fd_read(caller, 3, 400, 1, 500) == wasi_fs.EISDIR
    assert fs.fd_read(caller, 9, 400, 1, 500) == wasi_fs.EBADF


def test_close_semantics_for_stdio_preopen_files_and_unknown_fds():
    # Kills E-wasi_fs-170, M-fs-close-stdio, M-fs-close-preopen-kept.
    fs, caller, _ = _fs()
    for std in (0, 1, 2):
        assert fs.fd_close(caller, std) == 0
    assert fs.fd_close(caller, 7) == wasi_fs.EBADF
    assert fs.fd_close(caller, 3) == 0 and fs.fd_prestat_get(caller, 3, 0) == wasi_fs.EBADF


def test_fdstat_layout_for_stdio_directory_and_file():
    # Kills E-wasi_fs-175/176/183, M-fs-rights-dir-write, M-fs-fdstat-inherit.
    fs, caller, _ = _fs()
    for std in (0, 1, 2):
        caller.memory.data[900:930] = b"\xff" * 30
        assert fs.fd_fdstat_get(caller, std, 900) == 0
        assert bytes(caller.memory.data[900:924]) == bytes(24) and caller.memory.data[924] == 0xFF
    assert fs.fd_fdstat_get(caller, 3, 900) == 0
    ftype, flags, base, inherit = struct.unpack_from("<BxHxxxxQQ", caller.memory.data, 900)
    assert (ftype, flags, base, inherit) == (wasi_fs.DIRECTORY, 0, wasi_fs.RIGHTS_DIR, wasi_fs.RIGHTS_FILE)
    assert not base & 64 and not inherit & 64
    _, fd = _open(fs, caller, "python314.zip")
    assert fs.fd_fdstat_get(caller, fd, 900) == 0
    assert struct.unpack_from("<BxHxxxxQQ", caller.memory.data, 900) == (
        wasi_fs.REGULAR_FILE, 0, wasi_fs.RIGHTS_FILE, 0)
    assert fs.fd_fdstat_get(caller, 9, 900) == wasi_fs.EBADF


def test_stdio_filestat_writes_exactly_64_bytes():
    # Kills E-wasi_fs-98.
    fs, caller, _ = _fs()
    caller.memory.data[800:870] = b"\xff" * 70
    assert fs.fd_filestat_get(caller, 1, 800) == 0
    assert bytes(caller.memory.data[800:864]) == bytes(64) and caller.memory.data[864] == 0xFF
    assert fs.fd_filestat_get(caller, 9, 800) == wasi_fs.EBADF


def test_prestat_name_only_for_the_preopen_and_long_enough_buffers():
    # Kills E-wasi_fs-203, M-fs-prestat-namelen.
    fs, caller, _ = _fs()
    assert fs.fd_prestat_get(caller, 3, 0) == 0 and _u32(caller, 4) == 4
    assert fs.fd_prestat_dir_name(caller, 4, 16, 4) == wasi_fs.EBADF
    assert fs.fd_prestat_dir_name(caller, 3, 16, 3) == wasi_fs.EINVAL


@pytest.mark.parametrize("method, args, stdio, opened, bad", [
    ("fd_advise", (0, 0, 0), wasi_fs.ESPIPE, 0, wasi_fs.EBADF),
    ("fd_sync", (), 0, 0, wasi_fs.EBADF),
    ("fd_datasync", (), 0, 0, wasi_fs.EBADF),
    ("fd_fdstat_set_flags", (0,), wasi_fs.ENOTSUP, wasi_fs.ENOTSUP, wasi_fs.EBADF),
    ("fd_filestat_set_size", (0,), wasi_fs.EINVAL, wasi_fs.EROFS, wasi_fs.EBADF),
    ("fd_filestat_set_times", (0, 0, 0), wasi_fs.EPERM, wasi_fs.EROFS, wasi_fs.EBADF),
    ("fd_pwrite", (0, 0, 0, 0), wasi_fs.ESPIPE, wasi_fs.EBADF, wasi_fs.EBADF),
    ("fd_tell", (600,), wasi_fs.ESPIPE, 0, wasi_fs.EBADF),
])
def test_every_fd_call_answers_for_stdio_open_and_unknown_fds(method, args, stdio, opened, bad):
    # Kills the fd_advise/fd_sync/fd_fdstat_set_flags/fd_filestat_set_*/fd_pwrite survivors.
    fs, caller, _ = _fs()
    _, fd = _open(fs, caller, "python314.zip")
    call = getattr(fs, method)
    assert [call(caller, s, *args) for s in (0, 1, 2)] == [stdio] * 3
    assert call(caller, fd, *args) == opened
    assert call(caller, 9, *args) == bad


@pytest.mark.parametrize("method, args", [
    ("path_create_directory", (3, 100, 1)), ("path_remove_directory", (3, 100, 1)),
    ("path_unlink_file", (3, 100, 1)), ("path_filestat_set_times", (3, 0, 100, 1, 0, 0, 0)),
    ("path_symlink", (100, 1, 3, 100, 1)), ("path_link", (3, 0, 100, 1, 3, 100, 1)),
    ("path_rename", (3, 100, 1, 3, 100, 1)),
])
def test_mutating_path_calls_are_read_only_errors(method, args):
    # Kills M-fs-refuse-success and the _refuse variants.
    fs, caller, _ = _fs()
    assert getattr(fs, method)(caller, *args) == wasi_fs.EROFS


def test_readlink_is_einval_for_real_paths_and_resolution_errors_otherwise():
    # Kills E-wasi_fs-283 and M-fs-readlink-success.
    fs, caller, _ = _fs()
    n = _put(caller, 100, b"python314.zip")
    assert fs.path_readlink(caller, 3, 100, n, 300, 64, 500) == wasi_fs.EINVAL
    n = _put(caller, 100, b"nope")
    assert fs.path_readlink(caller, 3, 100, n, 300, 64, 500) == wasi_fs.ENOENT


def test_writes_route_only_stdout_and_stderr_and_report_the_length():
    # Kills M-fs-write-any-fd (fd 0 / file fds) on top of the existing fd 3 check.
    fs, caller, writes = _fs()
    _put(caller, 300, b"hey")
    _put(caller, 400, struct.pack("<II", 300, 3))
    _, fd = _open(fs, caller, "python314.zip")
    for bad in (0, fd):
        assert fs.fd_write(caller, bad, 400, 1, 500) == wasi_fs.EBADF
    assert fs.fd_write(caller, 1, 400, 1, 500) == 0 and _u32(caller, 500) == 3
    assert writes == [(1, b"hey")]


def test_iovec_array_at_the_end_of_memory_is_read_exactly():
    # Kills E-wasi_fs-77 (reading 9 bytes per iovec): an over-read past linear memory traps the guest.
    fs, caller, _ = _fs()
    _, fd = _open(fs, caller, "python314.zip")
    end = len(caller.memory.data) - 8
    _put(caller, end, struct.pack("<II", 300, 2))
    assert fs.fd_read(caller, fd, end, 1, 500) == 0 and bytes(caller.memory.data[300:302]) == b"01"


def test_fd_numbers_start_after_the_preopen_even_when_it_is_closed():
    # Kills E-wasi_fs-267 (allocation starting at 3).
    fs, caller, _ = _fs()
    assert _open(fs, caller, "") == (0, 4)          # a second handle on /lib
    assert fs.fd_close(caller, 3) == 0
    assert _open(fs, caller, "python314.zip", dirfd=4) == (0, 5)


# ================================================================ sandbox.py

import inspect  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

from qe_arena import sandbox  # noqa: E402

LOOP = "def f(x):\n    while x == 1:\n        pass\n    return x\n"


def test_budgets_and_caps_are_pinned_literals():
    # Kills the FUEL_PER_CALL / MODULE_FUEL / FUEL_PER_SECOND / MAX_RESPONSE_BYTES / mem_mb off-by-ones: the
    # suite compares fuel against the module's own constants, so any budget value passed.
    assert sandbox.FUEL_PER_CALL == 100_000_000 and sandbox.MODULE_FUEL == 200_000_000
    assert sandbox.FUEL_PER_SECOND == 200_000_000 and sandbox.MAX_RESPONSE_BYTES == 4 * 1024 * 1024
    assert inspect.signature(sandbox.run_calls).parameters["mem_mb"].default == 256
    [r] = sandbox.run_calls(LOOP, "f", [[1]])
    assert (r.kind, r.fuel) == ("fuel", 100_000_000)


def test_results_that_never_ran_carry_zero_fuel():
    # Kills E-sandbox-155 (CallResult default fuel) and E-sandbox-384 (missing "f" defaults to 0).
    assert sandbox.CallResult("rejected").fuel == 0
    assert all(r.fuel == 0 for r in sandbox.run_calls("import os\ndef f():\n    return 1\n", "f", [[]]))
    [row] = sandbox._parse_results({"results": [{"k": "ok", "v": 1}]}, 1)
    assert (row.kind, row.value, row.fuel) == ("ok", 1, 0)


def test_parse_results_rejects_bool_fuel_and_drops_values_of_unsettled_rows():
    # Kills M-pr-fuel-bool and M-pr-ok-value-none.
    with pytest.raises(ValueError):
        sandbox._parse_results({"results": [{"k": "ok", "v": 1, "f": True}]}, 1)
    [row] = sandbox._parse_results({"results": [{"k": "fuel", "v": 5, "f": 3}]}, 1)
    assert (row.kind, row.value, row.fuel) == ("fuel", None, 3)


def test_a_tiny_legacy_timeout_still_gets_one_unit_of_fuel():
    # Kills E-sandbox-416 and M-rc-fuel-legacy-int (0 fuel would never start, 2 is not the floor).
    [r] = sandbox.run_calls(LOOP, "f", [[1]], per_call_timeout_s=1e-12)
    assert (r.kind, r.fuel) == ("fuel", 1)


def test_a_failing_worker_is_retried_exactly_once(monkeypatch):
    # Kills E-sandbox-424 (three attempts).
    calls = []
    monkeypatch.setattr(sandbox, "_dispatch", lambda payload: calls.append(payload) or "exited")
    results = sandbox.run_calls("def f():\n    return 1\n", "f", [[]])
    assert [r.kind for r in results] == ["infra"] and len(calls) == 2


def test_runtime_root_honours_the_environment(monkeypatch):
    # Kills M-runtime-env-ignored.
    monkeypatch.setenv("QE_ARENA_WASI_ROOT", "/somewhere/else")
    assert sandbox.runtime_root() == Path("/somewhere/else")


@pytest.mark.parametrize("value, size", [(None, "detected"), ("3", 3), ("0", 1), ("-4", 1), ("many", "detected")])
def test_pool_size_parsing(monkeypatch, value, size):
    # Kills E-sandbox-307/309 and M-pool-size-floor. Without a usable override the cgroup-derived size applies.
    monkeypatch.setattr(sandbox, "_DETECTED", [2])
    if value is None:
        monkeypatch.delenv("QE_ARENA_WASI_WORKERS", raising=False)
    else:
        monkeypatch.setenv("QE_ARENA_WASI_WORKERS", value)
    assert sandbox._pool_size() == (2 if size == "detected" else size)


def _reader():
    r, w = os.pipe()
    worker = object.__new__(sandbox._Worker)
    worker.buf = bytearray()
    worker.proc = types.SimpleNamespace(stdout=os.fdopen(r, "rb", buffering=0))
    return worker, w


def test_worker_read_splits_lines_and_reports_eof_garbage_and_timeouts():
    # Kills E-sandbox-264..276 (line framing, deadline arithmetic, select loop) and M-read-bad-json-dict.
    worker, w = _reader()
    os.write(w, b'{"a": 1}\n{"b": 2}\n\nnot json\n{"c"')
    deadline = time.monotonic() + 5
    assert worker.read(deadline) == {"a": 1}
    assert worker.read(deadline) == {"b": 2}
    assert worker.read(deadline) == "bad response"     # empty line
    assert worker.read(deadline) == "bad response"     # garbage line
    threading.Timer(0.3, lambda: os.write(w, b': 3}\n')).start()
    assert worker.read(time.monotonic() + 0.9) == {"c": 3}   # completes across chunks inside the deadline
    t = time.monotonic()
    assert worker.read(time.monotonic() + 0.2) == "timeout"
    assert time.monotonic() - t < 0.9                         # not a second late (E-sandbox-272 0 -> -1)
    os.close(w)
    assert worker.read(time.monotonic() + 5) == "exited"


def test_worker_read_overflow_is_strictly_beyond_the_cap(monkeypatch):
    # Kills E-sandbox-280 (> -> >=) and M-read-no-overflow.
    monkeypatch.setattr(sandbox, "MAX_RESPONSE_BYTES", 10)
    worker, w = _reader()
    os.write(w, b'{"a": 12}\n')          # exactly 10 bytes
    assert worker.read(time.monotonic() + 5) == {"a": 12}
    os.write(w, b'{"a": 123}\n')         # 11 bytes
    assert worker.read(time.monotonic() + 5) == "overflow"


def test_worker_start_requires_a_ready_handshake(monkeypatch):
    # Kills E-sandbox-253 (and -> or): a worker that answers {"ready": false} must not join the pool.
    real = subprocess.Popen

    def fake(cmd, **kw):
        return real([sys.executable, "-c", "print('{\"ready\": false}', flush=True); import time; time.sleep(30)"], **kw)
    monkeypatch.setattr(sandbox.subprocess, "Popen", fake)
    with pytest.raises(RuntimeError):
        sandbox._Worker()


def test_worker_close_kills_a_live_process():
    # Kills E-sandbox-292 (close skipping a live process).
    worker = object.__new__(sandbox._Worker)
    worker.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    worker.close()
    assert worker.proc.poll() is not None


def _run_with_deadline(fn, seconds=120):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("v", fn()), daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "pool operation hung"
    return box.get("v")


def test_pool_accounting_through_warm_dispatch_failure_and_shutdown(monkeypatch):
    # Kills E-sandbox-314/319/323/328/340/352/357/365, M-acquire-*, M-dispatch-error-healthy.
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "1")
    sandbox.shutdown()
    assert sandbox._LIVE[0] == 0 and not sandbox._IDLE
    _run_with_deadline(sandbox.warm)
    assert sandbox._LIVE[0] == 1 and len(sandbox._IDLE) == 1
    first = sandbox._IDLE[0]
    assert [r.value for r in sandbox.run_calls("def f(x):\n    return x\n", "f", [[4]])] == [4]
    assert sandbox._IDLE == [first] and first.proc.poll() is None    # a healthy worker is reused

    held = sandbox._acquire(time.monotonic() + 60)                    # pool of 1 is now fully checked out
    got = {}
    waiter = threading.Thread(target=lambda: got.setdefault("w", sandbox._acquire(time.monotonic() + 60)),
                              daemon=True)
    waiter.start()
    waiter.join(2)
    assert waiter.is_alive() and sandbox._LIVE[0] == 1                # no over-commit beyond the pool
    sandbox._release(held, True)
    waiter.join(30)
    assert got["w"] is held
    sandbox._release(held, False)                                     # unhealthy: closed and uncounted
    assert sandbox._LIVE[0] == 0 and held.proc.poll() is not None

    monkeypatch.setattr(sandbox._Worker, "request", lambda self, payload, deadline: {"error": "boom"})
    assert sandbox._dispatch({}) == {"error": "boom"}
    assert sandbox._LIVE[0] == 0 and not sandbox._IDLE                # an error response retires the worker
    monkeypatch.undo()

    class Broken:
        def __init__(self):
            raise RuntimeError("no start")
    monkeypatch.setattr(sandbox, "_Worker", Broken)
    with pytest.raises(RuntimeError):
        sandbox._acquire(time.monotonic() + 60)
    assert sandbox._LIVE[0] == 0 and sum(sandbox._INFLIGHT.values()) == 0  # a failed start frees its slot
    monkeypatch.undo()
    sandbox.shutdown()
    assert sandbox._LIVE[0] == 0 and not sandbox._IDLE


# ================================================================ tasks.py (settled handling, generation, grading)

import json  # noqa: E402

from qe_arena import mutation, tasks  # noqa: E402
from qe_arena.tasks import make_task  # noqa: E402

CR = sandbox.CallResult


def test_generation_constants_are_pinned_literals():
    # Kills FAULTS / MIN_SUBTLE / MIN_REVEALING_D1 / MAX_SHOWN / HIDDEN_INPUTS off-by-ones that the generator
    # contract tests read back symbolically (tests compare with tasks.MIN_REVEALING_D1 itself).
    assert tasks.FAULTS == {1: 1, 2: 2, 3: 3} and tasks.MIN_SUBTLE == {1: 0, 2: 1, 3: 2}
    assert tasks.MIN_REVEALING_D1 == 3 and tasks.MAX_SHOWN == 200 and tasks.HIDDEN_INPUTS == 48
    assert tasks.RESOURCE_EXCEPTIONS == ("MemoryError", "RecursionError")


def test_clip_boundary():
    # Kills E-tasks-74 (<= -> <, slice shifts) and E-tasks-70 via behaviour, M-clip-none, M-result-text-exc-unclipped.
    assert tasks._clip("x" * 200) == "x" * 200
    assert tasks._clip("x" * 201) == "x" * 200 + "... (201 chars)"
    assert tasks._result_text(CR("exc", "E" * 300)) == "raises " + "E" * 200 + "... (300 chars)"
    assert tasks._result_text(CR("ok", "y" * 300)).endswith("... (302 chars)")


@pytest.mark.parametrize("result, settled", [
    (CR("ok", None), True), (CR("exc", "ValueError"), True), (CR("exc", "KeyError"), True),
    (CR("exc", "MemoryError"), False), (CR("exc", "RecursionError"), False),
    (CR("fuel"), False), (CR("resource"), False), (CR("crash"), False), (CR("skipped"), False), (CR("infra"), False),
    (CR("timeout"), False), (CR("rejected"), False),
])
def test_settled_matrix(result, settled):
    # Kills the _settled engine/manual survivors (fuel/resource counted as settled, resource names dropped).
    assert tasks._settled(result) is settled
    assert tasks._usable([CR("ok", 1), result]) is settled
    assert (tasks._killing([CR("ok", 1)], [result]) == [0]) is (settled and not sandbox.compare(CR("ok", 1), result))
    assert (tasks._killing([result], [CR("ok", 2)]) == [0]) is (settled and not sandbox.compare(result, CR("ok", 2)))


def _counting_run_calls(monkeypatch, first_kinds):
    seen = []

    def fake(source, entry, calls, **kw):
        seen.append(len(calls))
        if len(seen) == 1:
            return [CR(k) for k in first_kinds]
        return [CR("ok", 0) for _ in calls]
    monkeypatch.setattr(tasks.sandbox, "run_calls", fake)
    return seen


@pytest.mark.parametrize("first, retried", [
    (["ok", "infra"], True), (["timeout", "ok"], True), (["infra", "infra"], True),
    (["fuel", "ok"], False), (["resource", "crash"], False),
])
def test_outputs_retries_only_transient_infra(monkeypatch, first, retried):
    # Kills M-outputs-no-retry / -retry-all / -retry-infra-only: one infra row retries the batch once,
    # deterministic limits never do.
    first = [k if k != "ok" else "ok" for k in first]
    seen = _counting_run_calls(monkeypatch, first)
    tasks._outputs("def f(x):\n    return 0\n", "f", [[1], [2]])
    assert len(seen) == (2 if retried else 1)


def test_subsets_must_each_fail_settled_on_some_input(monkeypatch):
    # Kills M-subsets-skip-largest / -singletons-only / -drop-usable and the _subsets_all_fail engine survivors.
    original = "def f(x):\n    a = x + 1\n    b = a - 1\n    return b * 2\n"
    sites = {s.description: s for s in mutation.enumerate_sites(original)}
    plus = next(s for d, s in sites.items() if "'+' -> '-'" in d)
    minus = next(s for d, s in sites.items() if "'-' -> '+'" in d)
    times = next(s for d, s in sites.items() if "'*' -> '//'" in d)
    inputs = [[0], [3], [8]]
    reference = sandbox.run_calls(original, "f", inputs)
    assert tasks._subsets_all_fail(original, [plus, times], "f", inputs, reference)
    # plus and minus cancel: fixing only `times` leaves them in and passes every hidden input
    assert not tasks._subsets_all_fail(original, [plus, minus, times], "f", inputs, reference)
    real = tasks._outputs
    monkeypatch.setattr(tasks, "_outputs", lambda src, e, calls: [CR("fuel")] + real(src, e, calls)[1:])
    assert not tasks._subsets_all_fail(original, [plus, times], "f", inputs, reference)


def test_pool_is_validated_and_restricts_the_corpus():
    # Kills M-make-pool-unchecked / M-make-pool-ignored and the make_task pool engine survivors.
    for bad in ((), ("no_such_function",)):
        with pytest.raises(ValueError):
            make_task("qe_bugfix-d1", 0, pool=bad)
    name = make_task("qe_bugfix-d1", 0).private["corpus"]
    other = next(n for n in sorted(tasks.corpus.ENTRIES) if n != name)
    for task_id in ("qe_bugfix-d1", "qe_mutant_kill-d1"):
        assert make_task(task_id, 0, pool=(other,)).private["corpus"] == other


@pytest.mark.parametrize("level", [2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_subtle_quota_and_fault_count_per_level(level, seed):
    # Kills M-faults-*, M-minsubtle-zero, M-bf-drop-reserve, M-bf-reserve-gt with literal levels.
    task = make_task(f"qe_bugfix-d{level}", seed)
    p = task.private
    assert len(p["sites"]) == level
    pool = dict.fromkeys([*mutation.enumerate_sites(p["original"]), *p.get("fault_sites", [])])  # real faults too
    cands = [s for s in pool if s.description in p["sites"]]
    import itertools  # noqa: PLC0415
    sites = next(list(c) for c in itertools.combinations(cands, level)
                 if _apply_or_none(p["original"], list(c)) == p["buggy"])
    subtle = 0
    for s in sites:
        out = sandbox.run_calls(mutation.apply(p["original"], [s]), p["entry"], p["inputs"])
        kills = tasks._killing(p["reference"], out)
        assert kills and tasks._usable(out)
        subtle += len(kills) <= 0.15 * 48
    assert subtle >= {2: 1, 3: 2}[level]


def _apply_or_none(src, sites):
    try:
        return mutation.apply(src, sites)
    except ValueError:
        return None


@pytest.mark.parametrize("seed", [*range(12), 19, 20])  # without the filter, seed 20 draws a fault one hidden call reveals
def test_d1_fault_revealed_by_at_least_three_distinct_hidden_calls_literal(seed):
    # Kills E-tasks-28 (MIN_REVEALING_D1 3 -> 2) and M-bf-min-revealing-count-raw: the existing test compares
    # with tasks.MIN_REVEALING_D1 itself.
    p = make_task("qe_bugfix-d1", seed).private
    out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    revealing = {json.dumps(p["inputs"][i], sort_keys=True) for i in tasks._killing(p["reference"], out)}
    assert len(revealing) >= 3


def test_expected_matching_coerces_nested_ints_but_never_bools():
    # Kills M-aat-no-list / -no-dict / -isinstance / -any-to-float and _as_actual_types engine survivors.
    ok = lambda v: CR("ok", v)  # noqa: E731
    assert tasks._expected_matches([1, [2]], ok([1.0, [2.0]]))
    assert tasks._expected_matches({"a": {"b": 3}}, ok({"a": {"b": 3.0}}))
    assert not tasks._expected_matches([True], ok([1.0]))
    assert not tasks._expected_matches("1", ok(1.0))
    assert not tasks._expected_matches([1, 2], ok([1.0]))
    assert not tasks._expected_matches({"a": 1}, ok({"b": 1.0}))
    assert tasks._expected_matches({"a": 1, "x": 2}, ok({"a": 1.0, "x": 2})) is True


def test_error_texts_are_truncated_to_their_documented_lengths(monkeypatch):
    # Kills E-sandbox-376/390/430 (truncation lengths of parse errors, row errors and infra messages).
    with pytest.raises(ValueError) as info:
        sandbox._parse_results({"x" * 500: 1}, 1)
    assert len(str(info.value)) == 200
    [row] = sandbox._parse_results({"results": [{"k": "crash", "e": "e" * 500}]}, 1)
    assert row.error == "e" * 200
    monkeypatch.setattr(sandbox, "_dispatch", lambda payload: "x" * 500)
    [r] = sandbox.run_calls("def f():\n    return 1\n", "f", [[]])
    assert r.kind == "infra" and len(r.error) == 200


def test_a_subset_that_is_equivalent_alone_fails_the_check():
    # Kills E-tasks-124 (1 -> 2: singletons skipped) as a unit, independent of the generator's own filter.
    original = "def f(x):\n    if x > 100:\n        return 1\n    return x * 2\n"
    sites = mutation.enumerate_sites(original)
    ge = next(s for s in sites if "'>' -> '>='" in s.description)
    times = next(s for s in sites if "'*' -> '//'" in s.description)
    inputs = [[0], [3], [8]]
    reference = sandbox.run_calls(original, "f", inputs)
    assert not tasks._subsets_all_fail(original, [ge, times], "f", inputs, reference)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", range(8))
def test_buggy_file_settles_on_every_hidden_and_shown_input(level, seed):
    # Kills E-tasks-214/225 and M-bf-drop-combined-usable / -shown-usable: a buggy file that loops or
    # crashes on a graded or shown input makes fuel, not behaviour, decide the task.
    p = make_task(f"qe_bugfix-d{level}", seed).private
    assert tasks._usable(sandbox.run_calls(p["buggy"], p["entry"], p["inputs"]))
    shown = [v["args"] for v in p["visible"]] + ([p["report_args"]] if p["report_args"] is not None else [])
    assert tasks._usable(sandbox.run_calls(p["buggy"], p["entry"], shown))
    assert all(tasks._settled(v["expected"]) for v in p["visible"])


def test_multi_fault_generation_consults_the_cancellation_check(monkeypatch):
    # Kills M-subsets-call-dropped (and call-site and/or swaps). Cancelling faults are rare on the current
    # corpus (0 of 120 bugfix tasks for seeds 0-39 change without the check), so a seeded contract test
    # cannot see it; vetoing every combination must make d2 ungeneratable.
    name = make_task("qe_bugfix-d2", 0).private["corpus"]
    seen = []

    def veto(original, sites, entry, inputs, reference):
        seen.append(len(sites))
        return False
    monkeypatch.setattr(tasks, "_subsets_all_fail", veto)
    with pytest.raises(RuntimeError):
        make_task("qe_bugfix-d2", 0, pool=(name,))
    assert seen and set(seen) == {2}


def test_visible_run_retries_a_transient_infra_failure(monkeypatch):
    # Kills M-run-visible-run-calls: one worker hiccup must not show the model FAILs on passing code.
    task = make_task("qe_bugfix-d1", 9)
    real = sandbox.run_calls
    state = {"n": 0}

    def flaky(source, entry, calls, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return [CR("infra", error="simulated") for _ in calls]
        return real(source, entry, calls, **kw)
    monkeypatch.setattr(tasks.sandbox, "run_calls", flaky)
    report = tasks.run_visible(task, task.private["original"])
    assert report.startswith(f"{len(task.private['visible'])}/{len(task.private['visible'])} visible tests passed")


def test_long_visible_values_are_clipped_in_run_tests():
    # The existing run_tests cap test returns 300,000 characters, which the sandbox rejects as too large
    # (a crash), so it never reaches _clip. 100,000 characters is under the value cap and must be clipped.
    task = make_task("qe_bugfix-d1", 2)
    huge = f"def {task.private['entry']}(*args):\n    return '9' * 100000\n"
    report = tasks.run_visible(task, huge)
    assert len(report) < 2000 and "(100002 chars)" in report


def test_expected_matching_never_raises_on_mismatched_container_types():
    # Kills E-tasks-349 (and -> or): a list or string expected against a dict result raised AttributeError,
    # which _expected_matches does not catch ("never raises" in its docstring).
    ok = lambda v: CR("ok", v)  # noqa: E731
    assert tasks._expected_matches([1], ok({"a": 1.0})) is False
    assert tasks._expected_matches("a", ok({"a": 1.0})) is False
    assert tasks._expected_matches({"a": 1}, ok([1.0])) is False


# ================================================================ HIGH gaps on the merged head

def test_budgets_harness_and_memory_are_pinned_literals():
    from qe_arena import wasi_worker  # noqa: PLC0415
    assert (wasi_worker.STARTUP_FUEL, wasi_worker.HARNESS_FUEL, wasi_worker.BACKSTOP_S) == (3_000_000_000, 20_000_000, 60.0)
    assert sandbox.DEFAULT_MEM_MB == 256 and sandbox.MAX_VALUE_BYTES == 131072 and sandbox.MAX_RESULT_BYTES == 983040


@pytest.mark.parametrize("mutant", [
    "def f(x):\n    while True:\n        pass\n",                    # fuel
    "def f(x):\n    return len([0] * 10 ** 9)\n",                     # resource (memory)
    "def f(x):\n    return f(x)\n",                                  # resource (recursion)
    "def f(x):\n    s = []\n    for _ in range(100000):\n        s = [s]\n    return str(s)\n",  # wasm stack
])
def test_a_fuel_or_resource_mutant_outcome_is_never_a_kill(mutant):
    # HIGH (b): the existing test only simulated "timeout". Real fuel and resource outcomes from the WASI
    # sandbox must earn nothing, for an exact expected value or not.
    original = "def f(x):\n    return x + 1\n"
    task = tasks.Task(task_id="qe_mutant_kill-d1", family="qe_mutant_kill", difficulty=1, seed=0, prompt="",
                      files={}, private={"original": original, "entry": "f",
                                         "mutants": [{"source": mutant, "site": "", "random_kill_rate": 0.5}]})
    [out] = sandbox.run_calls(mutant, "f", [[3]], fuel=500_000_000)
    assert out.kind in ("fuel", "resource")
    assert tasks._killing([CR("ok", 4)], [out]) == []
    assert tasks.grade_mutant_kill(task, {"tests": [{"args": [3], "expected": 4}]}) == 0.0


class _FakeWorker:
    def close(self):
        pass


def test_warm_starts_exactly_the_pool_and_returns_promptly(monkeypatch):
    # HIGH (c): an off-by-one in warm() (pool_size + 1 acquisitions) waits for a slot that never frees.
    import collections  # noqa: PLC0415
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "2")
    monkeypatch.setattr(sandbox, "_IDLE", [])
    monkeypatch.setattr(sandbox, "_LIVE", [0])
    monkeypatch.setattr(sandbox, "_WAITING", collections.deque())
    monkeypatch.setattr(sandbox, "_INFLIGHT", collections.Counter())
    monkeypatch.setattr(sandbox, "_Worker", _FakeWorker)
    monkeypatch.setattr(sandbox, "WORKER_STARTUP_S", 1.0)  # an extra acquisition would time out within ~2 s
    t = time.monotonic()
    sandbox.warm()
    assert time.monotonic() - t < 1.0
    assert sandbox._LIVE[0] == len(sandbox._IDLE) == 2 and sum(sandbox._INFLIGHT.values()) == 0
    sandbox.warm()                                          # warming a warm pool starts nothing more
    assert sandbox._LIVE[0] == len(sandbox._IDLE) == 2
