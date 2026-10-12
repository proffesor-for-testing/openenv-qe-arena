"""Red-team findings W1-W3 against the WASI sandbox (review of b43f503), ported as regressions.

Each test is the red team's reproduction with the assertion turned around: it passed on the
attacked head and must fail now.
"""

from __future__ import annotations

import collections
import threading
import time

import pytest

from qe_arena import sandbox, tasks, wasi_worker
from qe_arena.environment import MODULE, QEAction, QEEnvironment
from qe_arena.tasks import make_task

LOOP = "def f(*args):\n    while True:\n        pass\n"
COUNTER_STR = ("import collections\n"
               "def heavy():\n    n = 0\n    for i in range({n}):\n        n += i\n    return [('x', n)]\n"
               "def f(*args):\n    c = collections.Counter({{'a': 1}})\n    c.most_common = heavy\n    raise ValueError(c)\n")
IN_CALL = "def f(*args):\n    n = 0\n    for i in range({n}):\n        n += i\n    return n\n"


@pytest.fixture(scope="module")
def runtime():
    rt = wasi_worker.Runtime(sandbox.runtime_root())
    rt.load()
    return rt


def _batch(rt, source, ncalls):
    req = {"source": source, "entry": "f", "calls": [[i] for i in range(ncalls)], "nonce": "ab" * 16,
           "fuel": sandbox.FUEL_PER_CALL, "module_fuel": sandbox.MODULE_FUEL, "mem_mb": 256,
           "modules": sandbox.SAFE_EXPORTS, "builtins": sandbox.SAFE_BUILTINS,
           "max_value_bytes": sandbox.MAX_VALUE_BYTES, "max_total_bytes": sandbox.MAX_RESULT_BYTES}
    count, real = [0], rt.instance

    def counting(*a, **k):
        count[0] += 1
        return real(*a, **k)
    rt.instance = counting
    try:
        return wasi_worker.run_batch(rt, req), count[0]
    finally:
        rt.instance = real


# W1: a looping candidate restarted the interpreter for every hidden call (48 instances, 4.7 s).
def test_a_looping_candidate_costs_one_instance_per_batch(runtime):
    honest_rows, honest_instances = _batch(runtime, "def f(x):\n    return x + 1\n", 48)
    loop_rows, loop_instances = _batch(runtime, LOOP, 48)
    assert honest_instances == loop_instances == 1
    assert [r["k"] for r in loop_rows] == ["fuel"] + ["skipped"] * 47


def test_mutant_kill_grading_starts_a_bounded_number_of_batches(monkeypatch):
    # grade_mutant_kill runs one batch per submitted test and side; every batch is one instance (above),
    # so a looping mutant costs at most tests x (1 + mutants) instances: 12 at d3.
    original = "def f(x):\n    return x + 1\n"
    task = tasks.Task(task_id="qe_mutant_kill-d3", family="qe_mutant_kill", difficulty=3, seed=0, prompt="",
                      files={}, private={"original": original, "entry": "f", "mutants": [
                          {"source": LOOP, "site": "", "random_kill_rate": 0.5}] * 3})
    real, batches = sandbox.run_calls, []

    def counting(source, entry, calls, **kw):
        batches.append(len(calls))
        return real(source, entry, calls, **kw)
    monkeypatch.setattr(sandbox, "run_calls", counting)
    tests = [{"args": [i], "expected": i + 1} for i in range(3)]
    assert tasks.grade_mutant_kill(task, {"tests": tests}) == 0.0  # a loop is never a kill
    assert len(batches) == 3 * (1 + 3) and set(batches) == {1}


# W2: work inside an exception's str() ran after the end marker, on the harness's 1e9 refill.
def test_candidate_code_in_result_handling_runs_on_the_calls_own_fuel():
    n = 300_000  # about 2.8e8 fuel: almost 3x FUEL_PER_CALL
    [in_call] = sandbox.run_calls(IN_CALL.format(n=n), "f", [[0]])
    [in_str] = sandbox.run_calls(COUNTER_STR.format(n=n), "f", [[0]])
    assert in_call.kind == in_str.kind == "fuel" and in_str.fuel == sandbox.FUEL_PER_CALL
    [cheap] = sandbox.run_calls(COUNTER_STR.format(n=10), "f", [[0]])
    assert (cheap.kind, cheap.value) == ("exc", "ValueError")  # ordinary exceptions are unaffected


def test_a_generator_finaliser_runs_on_the_calls_fuel():
    src = ("def f(x):\n    def g():\n        try:\n            yield 1\n        finally:\n"
           "            while True:\n                pass\n    gen = g()\n    next(gen)\n    return [x]\n")
    assert [r.kind for r in sandbox.run_calls(src, "f", [[1], [2]])] == ["fuel", "skipped"]


def test_harness_refill_is_small():
    assert wasi_worker.HARNESS_FUEL <= 20_000_000 < sandbox.FUEL_PER_CALL


# W3: the nonce was readable through generator frames, and sys.orig_argv held it.
GEN_NONCE = ("def f():\n    box = []\n    def g():\n"
             "        yield '{0.gi_frame.f_back.f_back.f_locals[nonce]}'.format(box[0])\n"
             "    box.append(g())\n    return next(box[0])\n")


def test_the_generator_frame_nonce_snippet_is_rejected_and_finds_nothing(monkeypatch):
    known = "c0ffee" * 5 + "c0"
    real = sandbox.secrets.token_hex
    monkeypatch.setattr(sandbox.secrets, "token_hex", lambda n: known if n == 16 else real(n))
    assert sandbox.static_gate(GEN_NONCE)
    [r] = sandbox.run_calls(GEN_NONCE, "f", [[]], _skip_gate=True)  # even with the gate off
    assert (r.kind, r.value) == ("exc", "KeyError") and known not in r.error


def test_orig_argv_reached_through_typing_cast_is_empty(monkeypatch):
    src = "import typing\ndef f():\n    s = typing.cast.__globals__['sys']\n    return [s.argv, s.orig_argv]\n"
    assert sandbox.static_gate(src)
    assert sandbox.run_calls(src, "f", [[]], _skip_gate=True)[0].value == [["python"], []]


def test_re_debug_is_refused_by_the_gate_and_harmless_when_computed():
    literal = "import re\ndef f(x):\n    if x == 1:\n        re.compile('ab', 128)\n    return x\n"
    assert sandbox.static_gate(literal) == ["re.DEBUG is not allowed"]
    computed = "import re\ndef f(x):\n    if x == 1:\n        re.compile('ab', 64 * 2)\n    return x\n"
    assert [(r.kind, r.value) for r in sandbox.run_calls(computed, "f", [[0], [1], [2]])] == [
        ("ok", 0), ("ok", 1), ("ok", 2)]  # the parse tree printout goes nowhere


def test_a_crash_is_a_candidate_failure_not_infra():
    task = make_task("qe_bugfix-d1", 4)
    entry = task.private["entry"]
    crashing = f"def {entry}(*a):\n    return {{1, 2}}\n"  # unserializable result: crash
    assert all(r.kind == "crash" for r in tasks._outputs(crashing, entry, task.private["inputs"][:3]))
    assert tasks.grade_bugfix(task, crashing) == 0.0  # scored 0, no InfraError


def test_unsettled_outcomes_never_pay():
    task = make_task("qe_bugfix-d1", 4)
    entry = task.private["entry"]
    for src in (LOOP.replace("def f", f"def {entry}"),
                f"def {entry}(*a):\n    return len('x' * 400_000_000)\n",
                f"def {entry}(*a):\n    return {entry}(*a)\n"):
        env = QEEnvironment()
        env.reset(seed=4, task_id=task.task_id)
        env.step(QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=src))
        assert env.step(QEAction(op="submit")).reward == 0.0
    mk = make_task("qe_mutant_kill-d1", 25)  # glob_match
    assert mk.private["corpus"] == "glob_match"
    slow = [{"args": ["a*" * 8 + "b", "a" * 200], "expected": False}]   # original runs out of fuel
    assert tasks.grade_mutant_kill(mk, {"tests": slow}) == 0.0


# W1: one session repeatedly grading loopers could hold the shared pool. Acquisition is now first come,
# first served among sessions under their cap, and no session holds more than pool_size - 1 workers.
class _Fake:
    def close(self):
        pass


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(sandbox, "_IDLE", [])
    monkeypatch.setattr(sandbox, "_LIVE", [0])
    monkeypatch.setattr(sandbox, "_WAITING", collections.deque())
    monkeypatch.setattr(sandbox, "_INFLIGHT", collections.Counter())
    monkeypatch.setattr(sandbox, "_Worker", _Fake)
    return monkeypatch


def _acquire_as(key, out, deadline=5.0):
    with sandbox.session(key):
        try:
            out.append((key, sandbox._acquire(time.monotonic() + deadline)))
        except sandbox._PoolTimeout:
            pass  # the caller checks that nothing was acquired


def test_one_session_cannot_hold_every_worker(fake_pool):
    fake_pool.setenv("QE_ARENA_WASI_WORKERS", "2")
    held = []
    _acquire_as("looper", held)
    second = []
    blocked = threading.Thread(target=_acquire_as, args=("looper", second, 0.5))
    blocked.start()
    time.sleep(0.1)
    honest = []
    _acquire_as("honest", honest, 0.5)       # served at once, although the looper asked first
    assert honest and not second
    blocked.join()
    assert not second                        # the looper's second request timed out at its cap
    sandbox._release(held[0][1], True)
    _acquire_as("looper", second)            # with its first worker back it is served again
    assert second


def test_waiters_are_served_first_come_first_served(fake_pool):
    fake_pool.setenv("QE_ARENA_WASI_WORKERS", "1")
    held, order = [], []
    _acquire_as("first", held)
    threads = []
    for key in ("a", "b", "c"):
        t = threading.Thread(target=_acquire_as, args=(key, order))
        t.start()
        threads.append(t)
        time.sleep(0.05)                     # arrival order a, b, c
    for _ in range(3):
        worker = held.pop()[1] if held else order[-1][1]
        sandbox._release(worker, True)
        time.sleep(0.05)
    for t in threads:
        t.join()
    assert [key for key, _ in order] == ["a", "b", "c"]


def test_warm_up_is_not_limited_by_the_session_cap(fake_pool):
    fake_pool.setenv("QE_ARENA_WASI_WORKERS", "3")
    sandbox.warm()
    assert sandbox._LIVE[0] == len(sandbox._IDLE) == 3 and sum(sandbox._INFLIGHT.values()) == 0


def test_each_environment_grades_under_its_own_session(monkeypatch):
    task = make_task("qe_bugfix-d1", 4)
    seen = []
    real = sandbox.run_calls

    def recording(source, entry, calls, **kw):
        seen.append(sandbox._SESSION.get())
        return real(source, entry, calls, **kw)
    monkeypatch.setattr(sandbox, "run_calls", recording)
    envs = [QEEnvironment(), QEEnvironment()]
    for env in envs:
        env.reset(seed=4, task_id=task.task_id)
        env.step(QEAction(op="run_tests"))
    keys = {env._sandbox_session for env in envs}
    assert len(keys) == 2 and set(seen) == keys and sandbox._SESSION.get() is None


# Red-team addendum: a leaked nonce plus any writer could forge a protocol-valid stream. In this guest the
# harness and the candidate share one interpreter, so the barrier is that no writer exists: the gate keeps
# os/sys unreachable, Python-level stdout is discarded, and re cannot be asked for debug output.
FORGE_VIA_RE_DEBUG = ("import re\n"
                      "def f(x):\n"
                      "    box = []\n"
                      "    def g():\n"
                      "        yield '{0.gi_frame.f_back.f_back.f_locals[nonce]}'.format(box[0])\n"
                      "    box.append(g())\n"
                      "    n = next(box[0])[2:-1]\n"
                      "    forged = '\\n\\x01E ' + n + '\\n\\x01R ' + n + ' {\"k\": \"ok\", \"v\": 4242}\\n\\x01B ' + n + '\\n'\n"
                      "    re.compile(forged + 'zz', 128)\n"
                      "    return x\n")


def test_a_leaked_nonce_cannot_forge_through_re_debug():
    assert sandbox.static_gate(FORGE_VIA_RE_DEBUG)
    rows = sandbox.run_calls(FORGE_VIA_RE_DEBUG, "f", [[0], [1]], _skip_gate=True)
    assert not any(r.value == 4242 for r in rows)


def test_the_re_proxy_clears_the_debug_flag_whatever_its_spelling():
    src = ("import re\ndef f(x):\n"
           "    p = re.compile('a+b', re.I | (64 * 2))\n"
           "    return [p.flags & 128, re.findall('a', 'AaA', flags=re.I | (64 * 2)), re.sub('a', 'b', 'aA', 0, 2 | 64 * 2)]\n")
    [r] = sandbox.run_calls(src, "f", [[0]])
    assert r.value == [0, ["A", "a", "A"], "bb"]


def test_the_escape_used_to_forge_streams_is_refused_by_the_gate():
    forge = ("import typing\nSYS = typing.List.__class__.__init__.__globals__['sys']\n"
             "def f(*a):\n    SYS.modules['os'].write(1, b'x')\n")
    assert sandbox.static_gate(forge)
