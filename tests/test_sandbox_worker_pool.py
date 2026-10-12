"""Sandbox parent side: pinned budgets, result parsing, worker framing and pool accounting.

Added after mutation testing b43f503 (65 surviving sandbox.py mutants, all killed by this file)."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from qe_arena import sandbox

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
    # Kills E-sandbox-307/309 and M-pool-size-floor. Ported: without a usable override the pool size is
    # the cgroup-derived one (978a0e7), not a fixed 2.
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
                              daemon=True)  # ported: _acquire takes a deadline (bf66e50)
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
        def __init__(self, *args):  # ported: _Worker now receives the dispatch deadline
            raise RuntimeError("no start")
    monkeypatch.setattr(sandbox, "_Worker", Broken)
    with pytest.raises(RuntimeError):
        sandbox._acquire(time.monotonic() + 60)
    assert sandbox._LIVE[0] == 0 and sum(sandbox._INFLIGHT.values()) == 0  # a failed start frees its slot
    monkeypatch.undo()
    sandbox.shutdown()
    assert sandbox._LIVE[0] == 0 and not sandbox._IDLE


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


# ---------------------------------------------------------------- added on the merged head (team lead HIGH c, MEDIUM)

def test_budgets_harness_and_memory_are_pinned_literals():
    from qe_arena import wasi_worker  # noqa: PLC0415
    assert (wasi_worker.STARTUP_FUEL, wasi_worker.HARNESS_FUEL, wasi_worker.BACKSTOP_S) == (3_000_000_000, 20_000_000, 60.0)
    assert sandbox.DEFAULT_MEM_MB == 256 and sandbox.MAX_VALUE_BYTES == 131072 and sandbox.MAX_RESULT_BYTES == 983040


class _FakeWorker:
    def __init__(self, *args):  # _acquire passes the dispatch deadline
        pass

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
