"""WASI preview1 surface of the in-memory filesystem: ABI values, every fd_*/path_* answer, fd allocation.

Added after mutation testing b43f503: 105 surviving wasi_fs mutants, all killed by this file."""

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
