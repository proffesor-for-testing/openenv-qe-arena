"""Long-lived sandbox worker: runs candidate Python inside CPython-WASI under wasmtime fuel.

Started by ``qe_arena.sandbox`` as its own process (never inside the uvicorn process, so a wasmtime
fault or a host bug cannot take down /health). It reads one JSON request per line on stdin and
writes one JSON response per line. Each batch gets a fresh Store and Instance:

* every call is metered in wasmtime fuel (about one unit per wasm instruction), never wall time,
  so a loop ends as ``fuel`` at exactly its budget on every host and under any load;
* linear memory is capped (``resource`` when exceeded, as is wasm stack exhaustion);
* the guest sees no host filesystem: ``qe_arena/wasi_fs.py`` serves one read-only directory,
  ``/lib``, holding the reproducible stdlib zip from memory, with canonical metadata (wasmtime's
  own preopen leaked host inode numbers, link counts and timestamps, which moved fuel between a
  host and a container on the same machine); the environment holds only PYTHONHASHSEED=0; clocks
  and clock resolution are fixed or stepped, ``random_get`` is a seeded SHAKE-256 stream and
  ``poll_oneoff`` is unsupported, so the guest cannot observe the host;
* call boundaries and per-call fuel come from nonce-tagged markers the harness writes to fd 1
  (see ``Protocol``); anything else on fd 1 fails the batch closed;
* a 60 s wall-clock epoch deadline exists only as an infrastructure backstop (``infra``), which is
  never graded.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import wasmtime as wt
from wasmtime import _ffi as ffi

from .wasi_fs import SIGNATURES as FS_SIGNATURES, VirtualFS

HARNESS = (Path(__file__).resolve().parent / "guest_harness.py").read_text(encoding="utf-8")
STARTUP_FUEL = 3_000_000_000   # interpreter init plus allowlisted imports (measured 287,333,867)
HARNESS_FUEL = 1_000_000_000   # harness bookkeeping between calls, refilled at every boundary
BACKSTOP_S = 60.0              # wall clock per instance: infrastructure failure only, never graded
EPOCH_TICK_S = 0.1
FIXED_REALTIME_NS = 1_700_000_000 * 10**9
MAX_GUEST_STDOUT = 4 << 20
ENOTSUP = 58
RNG_SEED = b"qe-arena"
GUEST_ARGV = ["python", "-S", "-s", "-P", "-B", "-c", HARNESS]
GUEST_ENV = [("PYTHONHASHSEED", "0"), ("PYTHONDONTWRITEBYTECODE", "1")]

I32 = wt.ValType.i32()
I64 = wt.ValType.i64()


class ProtocolError(Exception):
    """The guest's fd 1 carried something other than this run's markers, in order."""


class Protocol:
    """Parses the harness's fd 1 stream and drives fuel at call boundaries.

    Order is enforced: [B E R] for the module body, then [B E R] per call; a probe writes a single R.
    Every marker must carry this run's nonce. Violations raise ProtocolError, which the worker turns
    into a crash for the whole batch: nothing on fd 1 can forge a result or move fuel accounting.
    """

    def __init__(self, nonce: str, budgets: list[int], get_fuel: Callable[[], int],
                 set_fuel: Callable[[int], None]) -> None:
        self.begin = b"\x01B " + nonce.encode("ascii")
        self.end = b"\x01E " + nonce.encode("ascii")
        self.row_head = b"\x01R " + nonce.encode("ascii") + b" "
        self.budgets = budgets
        self.get_fuel, self.set_fuel = get_fuel, set_fuel
        self.state = "start"
        self.slot = 0
        self.budget_now = 0
        self.used: list[int] = []
        self.rows: list[dict] = []
        self.buf = bytearray()
        self.total = 0

    @property
    def in_call(self) -> bool:
        return self.state == "in"

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        if self.total > MAX_GUEST_STDOUT:
            raise ProtocolError("guest stdout over its cap")
        self.buf += data
        while True:
            nl = self.buf.find(b"\n")
            if nl < 0:
                return
            line = bytes(self.buf[:nl])
            del self.buf[:nl + 1]
            self._line(line)

    def _line(self, line: bytes) -> None:
        if line == self.begin and self.state in ("start", "ready"):
            if self.slot >= len(self.budgets):
                raise ProtocolError("more calls than budgets")
            self.budget_now = self.budgets[self.slot]
            self.state = "in"
            self.set_fuel(self.budget_now)
        elif line == self.end and self.state == "in":
            self.used.append(self.budget_now - self.get_fuel())
            self.slot += 1
            self.state = "after"
            self.set_fuel(HARNESS_FUEL)
        elif line.startswith(self.row_head) and self.state in ("after", "start"):
            try:
                row = json.loads(line[len(self.row_head):].decode("utf-8", "surrogatepass"))
            except ValueError as exc:
                raise ProtocolError(f"bad row: {exc}") from exc
            if not isinstance(row, dict):
                raise ProtocolError("row is not an object")
            self.rows.append(row)
            self.state = "ready" if self.state == "after" else "done"
        else:
            raise ProtocolError(f"unexpected line in state {self.state}: {line[:40]!r}")


# --------------------------------------------------------------------------- engine (one per process)

def make_engine() -> wt.Engine:
    cfg = wt.Config()
    cfg.consume_fuel = True
    cfg.epoch_interruption = True
    cfg.cranelift_nan_canonicalization = True   # NaN bit patterns identical across hosts
    cfg.wasm_relaxed_simd_deterministic = True
    cfg.wasm_threads = False
    return wt.Engine(cfg)


def set_caller_fuel(caller: wt.Caller, amount: int) -> None:
    """wasmtime-py exposes fuel only on Store; inside a host call we must use the caller's context.
    This uses the private ``_ffi`` surface, so tests/test_sandbox.py pins it (wasmtime==47.0.1)."""
    err = ffi.wasmtime_context_set_fuel(caller._context(), amount)
    if err:
        raise wt.WasmtimeError._from_ptr(err)


def get_caller_fuel(caller: wt.Caller) -> int:
    out = ctypes.c_uint64(0)
    err = ffi.wasmtime_context_get_fuel(caller._context(), ctypes.byref(out))
    if err:
        raise wt.WasmtimeError._from_ptr(err)
    return out.value


@dataclass
class Instance:
    """Outcome of one guest instance."""
    protocol: Protocol
    status: str          # exit | fuel | resource | infra | protocol | trap
    detail: str = ""
    remaining: int = 0   # fuel left in the store when the instance stopped

    @property
    def call_fuel(self) -> int:
        """Fuel the interrupted call consumed (meaningful when it died inside a call)."""
        return max(0, self.protocol.budget_now - self.remaining) if self.protocol.in_call else 0


@dataclass
class Runtime:
    root: Path
    engine: wt.Engine = field(default_factory=make_engine)
    module: Any = None
    files: dict = field(default_factory=dict)

    def load(self) -> None:
        self.files = {"/lib/python314.zip": (self.root / "lib" / "python314.zip").read_bytes()}
        self.module = wt.Module.from_file(self.engine, str(self.root / "python.wasm"))
        threading.Thread(target=self._ticker, daemon=True).start()

    def _ticker(self) -> None:
        while True:
            time.sleep(EPOCH_TICK_S)
            self.engine.increment_epoch()

    def _linker(self, state: dict) -> wt.Linker:
        linker = wt.Linker(self.engine)
        linker.define_wasi()           # args, environ, proc_exit, sched_yield, sock_* (on stdio only)
        linker.allow_shadowing = True

        def on_write(caller, fd, data):
            if fd != 1:
                return 0               # stderr: swallowed
            protocol = state["protocol"]
            protocol.get_fuel = lambda: get_caller_fuel(caller)
            protocol.set_fuel = lambda amount: set_caller_fuel(caller, amount)
            try:
                protocol.feed(data)
            except ProtocolError as exc:
                state["violation"] = str(exc)
                raise wt.Trap("protocol violation") from exc
            return 0

        fs = VirtualFS(self.files, on_write)

        def clock_time_get(caller, clock_id, precision, out_ptr):
            if clock_id == 0:
                value = FIXED_REALTIME_NS
            else:
                state["mono_ns"] += 1000     # stepped, never real
                value = state["mono_ns"]
            caller.get("memory").write(caller, value.to_bytes(8, "little"), out_ptr)
            return 0

        def clock_res_get(caller, clock_id, out_ptr):
            caller.get("memory").write(caller, (1000).to_bytes(8, "little"), out_ptr)  # not the host's
            return 0

        def random_get(caller, buf, length):
            if length:
                state["rng"] += 1
                data = hashlib.shake_256(RNG_SEED + state["rng"].to_bytes(8, "little")).digest(length)
                caller.get("memory").write(caller, data, buf)
            return 0

        def poll_oneoff(caller, a, b, c, d):
            return ENOTSUP                   # no sleeping, no waiting on the host

        def sig(params, results):
            return wt.FuncType(params, results)
        ns = "wasi_snapshot_preview1"
        types = {"i32": I32, "i64": I64}
        for name, (params, results) in FS_SIGNATURES.items():
            linker.define_func(ns, name, sig([types[t] for t in params], [types[t] for t in results]),
                               getattr(fs, name), access_caller=True)
        linker.define_func(ns, "clock_time_get", sig([I32, I64, I32], [I32]), clock_time_get, access_caller=True)
        linker.define_func(ns, "clock_res_get", sig([I32, I32], [I32]), clock_res_get, access_caller=True)
        linker.define_func(ns, "random_get", sig([I32, I32], [I32]), random_get, access_caller=True)
        linker.define_func(ns, "poll_oneoff", sig([I32] * 4, [I32]), poll_oneoff, access_caller=True)
        return linker

    def instance(self, request: dict, budgets: list[int], mem_mb: int) -> Instance:
        store = wt.Store(self.engine)
        store.set_limits(memory_size=mem_mb << 20)
        store.set_fuel(STARTUP_FUEL)
        store.set_epoch_deadline(int(BACKSTOP_S / EPOCH_TICK_S))
        wasi = wt.WasiConfig()
        wasi.argv = GUEST_ARGV + [json.dumps(request, ensure_ascii=True, allow_nan=True)]
        wasi.env = GUEST_ENV
        store.set_wasi(wasi)
        protocol = Protocol(request["nonce"], budgets, lambda: 0, lambda amount: None)
        state = {"protocol": protocol, "mono_ns": 0, "rng": 0, "violation": ""}
        inst = self._run(store, state)
        try:
            inst.remaining = store.get_fuel()
        except wt.WasmtimeError:
            inst.remaining = 0
        return inst

    def _run(self, store: wt.Store, state: dict) -> Instance:
        protocol = state["protocol"]
        try:
            self._linker(state).instantiate(store, self.module).exports(store)["_start"](store)
            return Instance(protocol, "exit", "0")
        except wt.ExitTrap as exc:
            return Instance(protocol, "exit", str(exc.code))
        except wt.Trap as exc:
            if state["violation"]:
                return Instance(protocol, "protocol", state["violation"])
            code = exc.trap_code
            if code == wt.TrapCode.OUT_OF_FUEL:
                return Instance(protocol, "fuel")
            if code == wt.TrapCode.INTERRUPT:
                return Instance(protocol, "infra", "wall-clock backstop")
            if code == wt.TrapCode.STACK_OVERFLOW:
                return Instance(protocol, "resource", "wasm stack exhausted")
            return Instance(protocol, "trap", (exc.message or "")[:200])
        except wt.WasmtimeError as exc:
            if state["violation"]:
                return Instance(protocol, "protocol", state["violation"])
            return Instance(protocol, "trap", str(exc)[:200])


def _row(kind: str, value: Any = None, error: str = "", fuel: int = 0) -> dict:
    return {"k": kind, "v": value, "e": error, "f": fuel}


def run_batch(runtime: Runtime, req: dict) -> list[dict]:
    """Run every call; a trap inside call k ends that call and the batch resumes at k + 1 in a fresh
    instance, so one looping input never voids the others."""
    calls = req["calls"]
    n = len(calls)
    budgets = [int(req["fuel"])] * n
    results: list[dict | None] = [None] * n
    start = 0
    while start < n:
        guest_req = {key: req[key] for key in ("source", "entry", "calls", "nonce", "modules", "builtins",
                                                "max_value_bytes", "max_total_bytes")}
        guest_req["start"] = start
        inst = runtime.instance(guest_req, [int(req["module_fuel"])] + budgets[start:], int(req["mem_mb"]))
        proto = inst.protocol
        if inst.status == "protocol":
            return [_row("crash", None, f"sandbox protocol: {inst.detail}")] * n
        if not proto.rows:  # died in startup or inside the module body
            if inst.status == "fuel" and proto.slot == 0 and proto.in_call:
                return [_row("fuel", None, "module body: fuel budget exhausted", proto.budget_now)] * n
            kind = inst.status if inst.status in ("infra", "resource") else "crash"
            return [_row(kind, None, f"module: {inst.status} {inst.detail}".strip())] * n
        head, rows = proto.rows[0], proto.rows[1:]
        if head.get("k") != "ok":
            kind = "resource" if head.get("k") == "resource" else "crash"
            return [_row(kind, None, str(head.get("e", ""))[:200])] * n
        for j, r in enumerate(rows):
            used = proto.used[j + 1] if j + 1 < len(proto.used) else 0
            kind = r.get("k")
            results[start + j] = _row(kind, r.get("v") if kind in ("ok", "exc") else None,
                                      str(r.get("e", ""))[:200], used)
        done = start + len(rows)
        if done >= n:
            break
        if inst.status == "exit":
            for i in range(done, n):
                results[i] = _row("crash", None, "guest exited early")
            break
        # the instance died inside call `done` (or in the harness right after it)
        if inst.status == "fuel" and proto.in_call:
            results[done] = _row("fuel", None, "fuel budget exhausted", proto.budget_now)
        elif inst.status in ("resource", "infra"):
            results[done] = _row(inst.status, None, inst.detail, inst.call_fuel)
        else:
            results[done] = _row("crash", None, f"{inst.status}: {inst.detail}"[:200])
        start = done + 1
    return results  # type: ignore[return-value]


def probe(runtime: Runtime, req: dict) -> dict:
    guest_req = {"probe": req["probe"], "nonce": req["nonce"], "modules": req["modules"]}
    inst = runtime.instance(guest_req, [], int(req.get("mem_mb", 256)))
    if inst.protocol.rows:
        return inst.protocol.rows[0]
    return {"error": f"{inst.status} {inst.detail}"}


def main() -> int:
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)  # nothing but responses may reach the parent's pipe
    root = Path(os.environ["QE_ARENA_WASI_ROOT"])
    runtime = Runtime(root)
    runtime.load()

    def send(obj: dict) -> None:
        out.write(json.dumps(obj, allow_nan=True).encode("utf-8", "surrogatepass") + b"\n")

    try:
        send({"ready": True})
        for line in sys.stdin.buffer:
            send(_answer(runtime, line))
    except BrokenPipeError:
        pass  # the parent went away (server shutdown): nothing left to answer
    return 0


def _answer(runtime: Runtime, line: bytes) -> dict:
    try:
        req = json.loads(line)
    except ValueError:
        return {"error": "bad request"}
    try:
        if "probe" in req:
            return {"id": req.get("id"), "probe": probe(runtime, req)}
        return {"id": req.get("id"), "results": run_batch(runtime, req)}
    except Exception as exc:  # noqa: BLE001 - report, keep serving
        return {"id": req.get("id"), "error": f"{type(exc).__name__}: {exc}"[:300]}

if __name__ == "__main__":
    sys.exit(main())
