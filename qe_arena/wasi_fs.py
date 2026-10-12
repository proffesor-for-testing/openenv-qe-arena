"""A read-only, in-memory WASI preview1 filesystem for the sandbox guest.

wasmtime's own filesystem passes the host's metadata through to the guest: 64-bit inode numbers,
link counts, directory sizes and real access/modify/change times (reading the stdlib zip even
bumps its atime, so the guest could read the host clock). The interpreter stats the stdlib at
startup, those values become Python ints of host-dependent size, and the heap, and therefore fuel,
differed between a Linux host and a container on the same machine.

So the guest gets no host directory at all. This module answers every fd_* and path_* call from
memory: one preopened directory ``/lib`` holding one file (the reproducible stdlib zip), with
canonical metadata (fixed times, fixed inode numbers, nlink 1). Stdio fds 0-2 look exactly as
wasmtime presents unattached stdio (unknown filetype, all-zero stat, not a tty); fd 1 and fd 2
writes go to ``on_write``. Anything that would modify the filesystem is refused.

The functions follow the WASI preview1 ABI (wasi_snapshot_preview1); every pointer is checked
against linear memory by wasmtime's memory accessors, which raise on an out-of-bounds access.
"""

from __future__ import annotations

import posixpath
import struct
from dataclasses import dataclass
from typing import Callable

# errno
SUCCESS, EACCES, EBADF, EINVAL, EISDIR, ENOENT, ENOTDIR, ENOTSUP, EPERM, EROFS, ESPIPE = (
    0, 2, 8, 28, 31, 44, 54, 58, 63, 69, 70)
# filetype
UNKNOWN, DIRECTORY, REGULAR_FILE = 0, 3, 4
# oflags / rights
O_CREAT, O_DIRECTORY, O_EXCL, O_TRUNC = 1, 2, 4, 8
FDFLAG_APPEND = 1
RIGHT_FD_WRITE = 1 << 6
RIGHTS_DIR = 0x1FFFFFFF & ~(RIGHT_FD_WRITE)
RIGHTS_FILE = (1 << 1) | (1 << 2) | (1 << 5) | (1 << 7) | (1 << 21)  # read, seek, tell, advise, filestat_get
WHENCE_SET, WHENCE_CUR, WHENCE_END = 0, 1, 2

FIXED_TIME_NS = 1_700_000_000 * 10**9
PREOPEN = "/lib"


@dataclass
class _Open:
    path: str
    is_dir: bool
    pos: int = 0


class VirtualFS:
    def __init__(self, files: dict[str, bytes], on_write: Callable[[object, int, bytes], int]) -> None:
        """``files`` maps absolute guest paths under /lib to contents. ``on_write(caller, fd, data)``
        receives fd 1 and fd 2 writes and returns the errno."""
        self.files = dict(files)
        self.dirs = {PREOPEN}
        for path in self.files:
            parent = posixpath.dirname(path)
            while parent.startswith(PREOPEN) and parent not in self.dirs:
                self.dirs.add(parent)
                parent = posixpath.dirname(parent)
        self.inodes = {path: i + 2 for i, path in enumerate(sorted(self.dirs | set(self.files)))}
        self.fds: dict[int, _Open] = {3: _Open(PREOPEN, True)}
        self.on_write = on_write

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mem(caller):
        return caller.get("memory")

    def _read(self, caller, ptr: int, length: int) -> bytes:
        return bytes(self._mem(caller).read(caller, ptr, ptr + length))

    def _write(self, caller, ptr: int, data: bytes) -> None:
        self._mem(caller).write(caller, data, ptr)

    def _iovs(self, caller, iovs: int, count: int) -> list[tuple[int, int]]:
        raw = self._read(caller, iovs, 8 * count)
        return [struct.unpack_from("<II", raw, 8 * i) for i in range(count)]

    def _resolve(self, dirfd: int, raw: bytes) -> tuple[int, str]:
        base = self.fds.get(dirfd)
        if base is None:
            return EBADF, ""
        if not base.is_dir:
            return ENOTDIR, ""
        rel = raw.decode("utf-8", "surrogateescape")
        if rel.startswith("/"):
            return EPERM, ""
        path = posixpath.normpath(posixpath.join(base.path, rel)) if rel else base.path
        if path != PREOPEN and not path.startswith(PREOPEN + "/"):
            return EPERM, ""  # escapes the preopen (wasmtime refuses these too)
        if path not in self.files and path not in self.dirs:
            return ENOENT, ""
        return SUCCESS, path

    def _filestat(self, path: str | None) -> bytes:
        if path is None:  # stdio, as wasmtime presents unattached stdio
            return bytes(64)
        is_dir = path in self.dirs
        size = 0 if is_dir else len(self.files[path])
        return struct.pack("<QQBxxxxxxxQQQQQ", 0, self.inodes[path], DIRECTORY if is_dir else REGULAR_FILE,
                           1, size, FIXED_TIME_NS, FIXED_TIME_NS, FIXED_TIME_NS)

    def _stdio_or_badf(self, fd: int, if_stdio: int) -> int:
        return if_stdio if fd in (0, 1, 2) else EBADF

    # ------------------------------------------------------------------ fd_*
    def fd_write(self, caller, fd, iovs, iovs_len, nwritten_ptr):
        if fd not in (1, 2):
            return EBADF
        data = b"".join(self._read(caller, p, n) for p, n in self._iovs(caller, iovs, iovs_len))
        self._write(caller, nwritten_ptr, struct.pack("<I", len(data)))
        return self.on_write(caller, fd, data)

    def fd_read(self, caller, fd, iovs, iovs_len, nread_ptr):
        return self._read_into(caller, fd, iovs, iovs_len, nread_ptr, None)

    def fd_pread(self, caller, fd, iovs, iovs_len, offset, nread_ptr):
        if fd in (0, 1, 2):
            return ESPIPE
        return self._read_into(caller, fd, iovs, iovs_len, nread_ptr, offset)

    def _read_into(self, caller, fd, iovs, iovs_len, nread_ptr, offset):
        if fd == 0:  # stdin is empty
            self._write(caller, nread_ptr, struct.pack("<I", 0))
            return SUCCESS
        handle = self.fds.get(fd)
        if handle is None:
            return EBADF
        if handle.is_dir:
            return EISDIR
        data = self.files[handle.path]
        pos = handle.pos if offset is None else offset
        total = 0
        for ptr, length in self._iovs(caller, iovs, iovs_len):
            chunk = data[pos:pos + length]
            if chunk:
                self._write(caller, ptr, chunk)
            pos += len(chunk)
            total += len(chunk)
            if len(chunk) < length:
                break
        if offset is None:
            handle.pos = pos
        self._write(caller, nread_ptr, struct.pack("<I", total))
        return SUCCESS

    def fd_seek(self, caller, fd, offset, whence, newoffset_ptr):
        handle = self.fds.get(fd)
        if handle is None:
            return self._stdio_or_badf(fd, ESPIPE)
        if handle.is_dir:
            return EBADF
        size = len(self.files[handle.path])
        base = {WHENCE_SET: 0, WHENCE_CUR: handle.pos, WHENCE_END: size}.get(whence)
        if base is None or base + offset < 0:
            return EINVAL
        handle.pos = base + offset
        self._write(caller, newoffset_ptr, struct.pack("<Q", handle.pos))
        return SUCCESS

    def fd_tell(self, caller, fd, offset_ptr):
        handle = self.fds.get(fd)
        if handle is None:
            return self._stdio_or_badf(fd, ESPIPE)
        self._write(caller, offset_ptr, struct.pack("<Q", handle.pos))
        return SUCCESS

    def fd_close(self, caller, fd):
        if fd in (0, 1, 2):
            return SUCCESS
        return SUCCESS if self.fds.pop(fd, None) is not None else EBADF

    def fd_fdstat_get(self, caller, fd, buf):
        if fd in (0, 1, 2):
            self._write(caller, buf, bytes(24))
            return SUCCESS
        handle = self.fds.get(fd)
        if handle is None:
            return EBADF
        rights = RIGHTS_DIR if handle.is_dir else RIGHTS_FILE
        filetype = DIRECTORY if handle.is_dir else REGULAR_FILE
        self._write(caller, buf, struct.pack("<BxHxxxxQQ", filetype, 0, rights, RIGHTS_FILE if handle.is_dir else 0))
        return SUCCESS

    def fd_filestat_get(self, caller, fd, buf):
        if fd in (0, 1, 2):
            self._write(caller, buf, self._filestat(None))
            return SUCCESS
        handle = self.fds.get(fd)
        if handle is None:
            return EBADF
        self._write(caller, buf, self._filestat(handle.path))
        return SUCCESS

    def fd_prestat_get(self, caller, fd, buf):
        if fd != 3 or 3 not in self.fds:
            return EBADF
        self._write(caller, buf, struct.pack("<BxxxI", 0, len(PREOPEN)))
        return SUCCESS

    def fd_prestat_dir_name(self, caller, fd, path, path_len):
        if fd != 3 or 3 not in self.fds:
            return EBADF
        if path_len < len(PREOPEN):
            return EINVAL
        self._write(caller, path, PREOPEN.encode())
        return SUCCESS

    def fd_readdir(self, caller, fd, buf, buf_len, cookie, bufused_ptr):
        handle = self.fds.get(fd)
        if handle is None:
            return EBADF
        if not handle.is_dir:
            return ENOTDIR
        names = sorted(posixpath.basename(p) for p in self.dirs | set(self.files)
                       if posixpath.dirname(p) == handle.path and p != handle.path)
        entries = [(".", handle.path, DIRECTORY), ("..", posixpath.dirname(handle.path), DIRECTORY)]
        entries += [(n, posixpath.join(handle.path, n),
                     DIRECTORY if posixpath.join(handle.path, n) in self.dirs else REGULAR_FILE) for n in names]
        out = bytearray()
        for index in range(max(0, cookie), len(entries)):
            name, path, filetype = entries[index]
            encoded = name.encode("utf-8", "surrogateescape")
            out += struct.pack("<QQIBxxx", index + 1, self.inodes.get(path, 1), len(encoded), filetype) + encoded
            if len(out) >= buf_len:
                break
        data = bytes(out[:buf_len])
        if data:
            self._write(caller, buf, data)
        self._write(caller, bufused_ptr, struct.pack("<I", len(data)))
        return SUCCESS

    def fd_advise(self, caller, fd, offset, length, advice):
        return SUCCESS if fd in self.fds else self._stdio_or_badf(fd, ESPIPE)

    def fd_sync(self, caller, fd):
        return SUCCESS if fd in self.fds or fd in (0, 1, 2) else EBADF

    fd_datasync = fd_sync

    def fd_fdstat_set_flags(self, caller, fd, flags):
        return ENOTSUP if fd in self.fds or fd in (0, 1, 2) else EBADF

    def fd_filestat_set_size(self, caller, fd, size):
        return EROFS if fd in self.fds else self._stdio_or_badf(fd, EINVAL)

    def fd_filestat_set_times(self, caller, fd, atim, mtim, flags):
        return EROFS if fd in self.fds else self._stdio_or_badf(fd, EPERM)

    def fd_pwrite(self, caller, fd, iovs, iovs_len, offset, nwritten_ptr):
        return EBADF if fd not in (0, 1, 2) else ESPIPE

    # ------------------------------------------------------------------ path_*
    def path_open(self, caller, dirfd, dirflags, path, path_len, oflags, rights_base, rights_inheriting,
                  fdflags, fd_ptr):
        if oflags & (O_CREAT | O_TRUNC) or fdflags & FDFLAG_APPEND or rights_base & RIGHT_FD_WRITE:
            if dirfd not in self.fds:
                return EBADF
            return EACCES
        errno, resolved = self._resolve(dirfd, self._read(caller, path, path_len))
        if errno:
            return errno
        is_dir = resolved in self.dirs
        if oflags & O_DIRECTORY and not is_dir:
            return ENOTDIR
        fd = 4
        while fd in self.fds:
            fd += 1
        self.fds[fd] = _Open(resolved, is_dir)
        self._write(caller, fd_ptr, struct.pack("<I", fd))
        return SUCCESS

    def path_filestat_get(self, caller, dirfd, flags, path, path_len, buf):
        errno, resolved = self._resolve(dirfd, self._read(caller, path, path_len))
        if errno:
            return errno
        self._write(caller, buf, self._filestat(resolved))
        return SUCCESS

    def path_readlink(self, caller, dirfd, path, path_len, buf, buf_len, bufused_ptr):
        errno, _ = self._resolve(dirfd, self._read(caller, path, path_len))
        return errno or EINVAL  # nothing here is a symlink

    def _refuse(self, dirfd: int) -> int:
        return EROFS if dirfd in self.fds else EBADF

    def path_create_directory(self, caller, dirfd, path, path_len):
        return self._refuse(dirfd)

    def path_remove_directory(self, caller, dirfd, path, path_len):
        return self._refuse(dirfd)

    def path_unlink_file(self, caller, dirfd, path, path_len):
        return self._refuse(dirfd)

    def path_filestat_set_times(self, caller, dirfd, flags, path, path_len, atim, mtim, fst_flags):
        return self._refuse(dirfd)

    def path_symlink(self, caller, old, old_len, dirfd, new, new_len):
        return self._refuse(dirfd)

    def path_link(self, caller, old_fd, old_flags, old, old_len, new_fd, new, new_len):
        return self._refuse(new_fd)

    def path_rename(self, caller, fd, old, old_len, new_fd, new, new_len):
        return self._refuse(new_fd)


I32, I64 = "i32", "i64"
SIGNATURES: dict[str, tuple[list[str], list[str]]] = {
    "fd_advise": ([I32, I64, I64, I32], [I32]),
    "fd_close": ([I32], [I32]),
    "fd_datasync": ([I32], [I32]),
    "fd_fdstat_get": ([I32, I32], [I32]),
    "fd_fdstat_set_flags": ([I32, I32], [I32]),
    "fd_filestat_get": ([I32, I32], [I32]),
    "fd_filestat_set_size": ([I32, I64], [I32]),
    "fd_filestat_set_times": ([I32, I64, I64, I32], [I32]),
    "fd_pread": ([I32, I32, I32, I64, I32], [I32]),
    "fd_prestat_get": ([I32, I32], [I32]),
    "fd_prestat_dir_name": ([I32, I32, I32], [I32]),
    "fd_pwrite": ([I32, I32, I32, I64, I32], [I32]),
    "fd_read": ([I32, I32, I32, I32], [I32]),
    "fd_readdir": ([I32, I32, I32, I64, I32], [I32]),
    "fd_seek": ([I32, I64, I32, I32], [I32]),
    "fd_sync": ([I32], [I32]),
    "fd_tell": ([I32, I32], [I32]),
    "fd_write": ([I32, I32, I32, I32], [I32]),
    "path_create_directory": ([I32, I32, I32], [I32]),
    "path_filestat_get": ([I32, I32, I32, I32, I32], [I32]),
    "path_filestat_set_times": ([I32, I32, I32, I32, I64, I64, I32], [I32]),
    "path_link": ([I32, I32, I32, I32, I32, I32, I32], [I32]),
    "path_open": ([I32, I32, I32, I32, I32, I64, I64, I32, I32], [I32]),
    "path_readlink": ([I32, I32, I32, I32, I32, I32], [I32]),
    "path_remove_directory": ([I32, I32, I32], [I32]),
    "path_rename": ([I32, I32, I32, I32, I32, I32], [I32]),
    "path_symlink": ([I32, I32, I32, I32, I32], [I32]),
    "path_unlink_file": ([I32, I32, I32], [I32]),
}
