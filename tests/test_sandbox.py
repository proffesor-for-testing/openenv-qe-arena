"""Sandbox and oracle comparison. This is the reward function: tests think like an attacker.

The sandbox runs CPython 3.14 on WASI under wasmtime fuel in a worker process. Outcomes are
deterministic, so the old wall-clock and rlimit tests are now exact fuel and WASI-surface tests.
"""

from __future__ import annotations

import collections
import ctypes
import importlib.metadata
import json
import os
import pathlib
import signal
import statistics as pystats
import sys
import threading
import time

import pytest
import wasmtime as wt

from qe_arena import sandbox, wasi_fs, wasi_worker
from qe_arena.sandbox import FUEL_PER_CALL, MODULE_FUEL, CallResult, compare, run_calls, static_gate

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_wasi_runtime  # noqa: E402
import fingerprint  # noqa: E402

LINUX = sys.platform.startswith("linux")
CORPUS = ROOT / "qe_arena" / "corpus"

# Cross-host golden: identical on arm64 macOS, amd64 Linux and the amd64 arena image. Regenerate only
# when the runtime, guest harness, budgets, SAFE_EXPORTS or corpus change, and say why in the commit.
# CORPUS_FINGERPRINT covers the software-engineering entries; each other domain has its own pin, so a
# domain's new entries move only its own line (`python scripts/fingerprint.py` prints them all).
CORPUS_FINGERPRINT = ("f50dbb1dd3917498", 1_732_616_699, 2880)
DOMAIN_FINGERPRINTS = {
    "finance-economics": ("7046fade88fc5037", 179_580_773, 256),
    "industrial-physical-systems": ("5b764def6559613a", 167_521_565, 256),
}


def ok(value):
    return CallResult("ok", value)


def kinds(results):
    return [r.kind for r in results]


def outcome(results):
    return [(r.kind, r.value, r.fuel) for r in results]


# --------------------------------------------------------------------------- values

def test_values_round_trip_and_normalize():
    src = ("def f(x):\n"
           "    return {'t': (1, (2, 3)), 'b': True, 'n': None, 'f': x / 4, 's': 'é\\n', 'l': [x, -x]}\n")
    [r] = run_calls(src, "f", [[3]])
    assert r.kind == "ok"
    assert r.value == {"t": [1, [2, 3]], "b": True, "n": None, "f": 0.75, "s": "é\n", "l": [3, -3]}
    assert type(r.value["b"]) is bool and type(r.value["f"]) is float


def test_exceptions_report_type_name():
    src = ("def f(x):\n"
           "    if x == 0:\n        raise ValueError('zero')\n"
           "    if x == 1:\n        return {}[x]\n"
           "    if x == 2:\n        return 1 // 0\n"
           "    if x == 3:\n        return f(x)\n"
           "    return x\n")
    results = run_calls(src, "f", [[0], [1], [2], [3], [4]])
    assert [(r.kind, r.value) for r in results] == [
        ("exc", "ValueError"), ("exc", "KeyError"), ("exc", "ZeroDivisionError"), ("resource", None),
        ("ok", 4)]
    assert results[0].error == "ValueError: zero"


def test_unserializable_returns_are_crashes():
    src = ("def f(x):\n"
           "    if x == 0:\n        return {1, 2}\n"
           "    if x == 1:\n        return {1: 'int key'}\n"
           "    if x == 2:\n        return (i for i in range(3))\n"
           "    if x == 3:\n        return 10 ** 5000\n"
           "    if x == 4:\n        v = []\n        for _ in range(500):\n            v = [v]\n        return v\n"
           "    return f\n")
    results = run_calls(src, "f", [[i] for i in range(6)])
    assert [(r.kind, r.error) for r in results] == [("crash", "unserializable")] * 6


def test_oversize_value_is_a_crash_not_a_truncation():
    src = "def f(n):\n    return 'x' * n\n"
    results = run_calls(src, "f", [[10], [200_000], [10]])
    assert [(r.kind, r.error) for r in results] == [("ok", ""), ("crash", "output too large"), ("ok", "")]


def test_total_output_budget_is_enforced_per_call():
    src = "def f(n):\n    return 'y' * n\n"
    results = run_calls(src, "f", [[120_000]] * 10)
    assert kinds(results)[:8] == ["ok"] * 8
    assert [(r.kind, r.error) for r in results[8:]] == [("crash", "output too large")] * 2


def test_entry_missing_or_not_callable_crashes_every_call():
    assert kinds(run_calls("def g():\n    return 1\n", "f", [[], []])) == ["crash", "crash"]
    assert kinds(run_calls("f = 3\n", "f", [[]])) == ["crash"]


def test_module_level_failure_crashes_every_call():
    results = run_calls("x = 1 // 0\ndef f():\n    return 1\n", "f", [[], []])
    assert [(r.kind, r.error) for r in results] == [("crash", "module: ZeroDivisionError: division by zero")] * 2


def test_print_is_a_silent_noop():
    src = "def f(x):\n    print('hello', x, sep='-')\n    return x\n"
    assert [r.value for r in run_calls(src, "f", [[1], [2]])] == [1, 2]


def test_allowlisted_modules_work():
    src = ('import math, re, itertools, functools, collections, string, bisect, heapq, statistics, operator\n'
           'from decimal import Decimal\nfrom fractions import Fraction\nfrom typing import List, Optional\n'
           'def f(xs: List[int]) -> Optional[list]:\n'
           '    c = collections.Counter(xs)\n'
           '    return [math.isqrt(sum(xs)), re.findall(r"\\d", "a1b2"), list(itertools.accumulate(xs)),\n'
           '            functools.reduce(operator.mul, xs, 1), c.most_common(1)[0][0], string.digits[:3],\n'
           '            bisect.bisect_left(sorted(xs), 3), heapq.nsmallest(2, xs), statistics.median(xs),\n'
           '            str(Decimal("1.10") + Decimal("2.205")), str(Fraction(1, 3) + Fraction(1, 6)),\n'
           '            sorted(xs, key=operator.neg)[:2], operator.itemgetter(0, 2)(xs)]\n')
    [r] = run_calls(src, "f", [[[4, 1, 3, 3, 9]]])
    assert r.kind == "ok", r
    assert r.value == [4, ["1", "2"], [4, 5, 8, 11, 20], 324, 3, "012", 1, [1, 3], 3, "3.305", "1/2",
                       [9, 4], [4, 3]]


def test_args_are_deep_copied_per_call():
    src = "def f(xs):\n    xs.append(0)\n    xs[0].append(1)\n    return [len(xs), len(xs[0])]\n"
    shared = [[]]
    calls = [[shared], [shared], [shared]]
    results = run_calls(src, "f", calls)
    assert [r.value for r in results] == [[2, 1]] * 3
    assert shared == [[]]


def test_module_state_cannot_leak_expected_values_but_is_per_batch():
    src = "seen = []\ndef f(x):\n    seen.append(x)\n    return len(seen)\n"
    assert [r.value for r in run_calls(src, "f", [[1], [2]])] == [1, 2]
    assert [r.value for r in run_calls(src, "f", [[1], [2]])] == [1, 2]  # fresh process per batch


def test_calls_must_be_argument_lists():
    with pytest.raises(TypeError):
        run_calls("def f(x):\n    return x\n", "f", [{"x": 1}])
    assert run_calls("def f(x):\n    return x\n", "f", []) == []


# --------------------------------------------------------------------------- fuel (replaces wall-clock timeouts)

LOOP = "def f(x):\n    while x == 1:\n        pass\n    return x\n"


def test_a_looping_call_ends_at_exactly_its_budget_and_skips_the_rest():
    results = run_calls(LOOP, "f", [[0], [2], [1], [3], [4]])
    assert [(r.kind, r.value) for r in results] == [("ok", 0), ("ok", 2), ("fuel", None), ("skipped", None),
                                                     ("skipped", None)]
    assert results[2].fuel == FUEL_PER_CALL
    assert all(0 < r.fuel < 1_000_000 for r in results if r.kind == "ok")  # call + result row + cleanup


def test_a_trap_never_lets_later_calls_run_on_reconstructed_state():
    # Codex P3: calls after a trap used to run in a fresh instance, so [[0], [1], [2]] gave ok 1 for
    # the last call where [[0], [2]] gives ok 2. Later calls are skipped instead.
    src = "seen = []\ndef f(x):\n    seen.append(x)\n    if x == 1:\n        while True:\n            pass\n    return len(seen)\n"
    trapped = run_calls(src, "f", [[0], [1], [2]], fuel=1_000_000)
    clean = run_calls(src, "f", [[0], [2]], fuel=1_000_000)
    assert [(r.kind, r.value) for r in trapped] == [("ok", 1), ("fuel", None), ("skipped", None)]
    assert [(r.kind, r.value) for r in clean] == [("ok", 1), ("ok", 2)]
    assert not compare(trapped[2], trapped[2])  # skipped is never settled


def test_fuel_exhaustion_cannot_be_swallowed():
    src = ("def f(x):\n"
           "    while True:\n"
           "        try:\n            while True:\n                pass\n"
           "        except:\n            pass\n")
    assert outcome(run_calls(src, "f", [[0], [1]])) == [("fuel", None, FUEL_PER_CALL), ("skipped", None, 0)]


def test_c_level_work_is_metered_too():
    t = time.monotonic()
    results = run_calls("def f(x):\n    return sum(range(10 ** 13))\n", "f", [[0], [1]])
    assert outcome(results) == [("fuel", None, FUEL_PER_CALL), ("skipped", None, 0)]
    assert time.monotonic() - t < 60.0  # a liveness bound, not a speed claim


def test_module_level_infinite_loop_is_fuel_for_every_call():
    results = run_calls("while True:\n    pass\ndef f():\n    return 1\n", "f", [[], []])
    assert outcome(results) == [("fuel", None, MODULE_FUEL)] * 2


def test_budget_parameters():
    assert outcome(run_calls(LOOP, "f", [[1]], fuel=1_000_000)) == [("fuel", None, 1_000_000)]
    legacy = run_calls(LOOP, "f", [[1]], per_call_timeout_s=0.05)  # converted at FUEL_PER_SECOND
    assert outcome(legacy) == [("fuel", None, int(0.05 * sandbox.FUEL_PER_SECOND))]
    assert run_calls(LOOP, "f", [[0]], timeout_s=0.001)[0].kind == "ok"  # wall clock no longer decides


def test_outcomes_and_fuel_are_identical_across_runs_and_under_load():
    src = ("import collections\n"
           "def f(xs, k):\n"
           "    if k == 3:\n        while True:\n            pass\n"
           "    if k == 4:\n        raise KeyError(k)\n"
           "    return sorted(collections.Counter(x % k for x in xs).items())\n")
    calls = [[list(range(i, i + 30)), 1 + i % 5] for i in range(24) if i % 5 != 2] + [[[1, 2], 3], [[3], 1]]
    baseline = outcome(run_calls(src, "f", calls))
    assert [k for k, _, _ in baseline][-2:] == ["fuel", "skipped"] and {k for k, _, _ in baseline} == {
        "ok", "exc", "fuel", "skipped"}
    stop = threading.Event()

    def burn():
        while not stop.is_set():
            sum(i * i for i in range(10000))
    burners = [threading.Thread(target=burn, daemon=True) for _ in range(4)]
    for b in burners:
        b.start()
    try:
        runs = [outcome(run_calls(src, "f", calls)) for _ in range(5)]
    finally:
        stop.set()
    assert all(r == baseline for r in runs)


def test_corpus_fingerprint_is_pinned_across_hosts():
    assert fingerprint.corpus_fingerprint() == CORPUS_FINGERPRINT


def test_every_domain_corpus_fingerprint_is_pinned_across_hosts():
    domains = fingerprint.domains_with_entries()
    assert domains[0] == "software-engineering" and set(domains[1:]) == set(DOMAIN_FINGERPRINTS)
    for domain in domains[1:]:
        assert fingerprint.corpus_fingerprint(domain) == DOMAIN_FINGERPRINTS[domain], domain


def test_runtime_is_the_pinned_reproducible_build():
    assert fingerprint.runtime_digests() == wasi_worker.PINS
    assert json.loads(build_wasi_runtime.DEFAULT_PINS.read_text()) == wasi_worker.PINS


# Codex P7: base images by digest and wasmtime by hash, and the pins agree with each other.
def test_image_build_inputs_are_pinned():
    dockerfile = (ROOT / "Dockerfile").read_text()
    froms = [line.split()[1] for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert len(froms) == 2 and all("@sha256:" in image and len(image.split("@sha256:")[1]) == 64 for image in froms)
    assert "--require-hashes -r /build/requirements-wasmtime.lock" in dockerfile
    assert "--require-hashes -r /app/requirements-wasmtime.lock" in dockerfile
    hashed = (ROOT / "requirements-wasmtime.lock").read_text()
    locked = [ln for ln in (ROOT / "requirements.lock").read_text().splitlines() if ln.startswith("wasmtime==")]
    assert locked == ["wasmtime==47.0.1"] and "wasmtime==47.0.1 \\" in hashed
    assert hashed.count("--hash=sha256:") >= 2
    assert importlib.metadata.version("wasmtime") == "47.0.1"


# Codex P6: the worker used whatever runtime its path pointed at; only the tests checked identity.
def test_a_worker_refuses_a_runtime_that_is_not_the_pinned_build(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    (root / "lib").mkdir(parents=True)
    (root / "python.wasm").symlink_to(sandbox.runtime_root() / "python.wasm")
    good = (sandbox.runtime_root() / "lib" / "python314.zip").read_bytes()
    (root / "lib" / "python314.zip").write_bytes(good[:-1] + bytes([good[-1] ^ 1]))  # one bit flipped
    monkeypatch.setenv("QE_ARENA_WASI_ROOT", str(root))
    with pytest.raises(RuntimeError, match="python314.zip sha256 .* is not the pinned"):
        sandbox._Worker()


def test_runtime_load_checks_the_bytes_it_serves(tmp_path):
    runtime = wasi_worker.Runtime(sandbox.runtime_root())
    with pytest.raises(wasi_worker.RuntimeIdentityError, match="python.wasm"):
        runtime.load(dict(wasi_worker.PINS, **{"python.wasm": "0" * 64}))
    assert runtime.module is None


# --------------------------------------------------------------------------- memory and stack

def test_allocation_beyond_the_cap_is_resource():
    results = run_calls("def f(x):\n    return len([0] * 10 ** 9)\n", "f", [[0], [1]])
    assert [r.kind for r in results] == ["resource"] * 2


def test_gradual_memory_growth_is_resource_at_a_deterministic_point():
    src = "def f(x):\n    s = 'a'\n    while True:\n        s = s + s\n"
    a, b = (run_calls(src, "f", [[0]], fuel=2_000_000_000)[0] for _ in range(2))
    assert a.kind == "resource" and a.fuel == b.fuel and a.fuel > 0


def test_memory_cap_follows_mem_mb():
    src = "def f(n):\n    return len([0] * n)\n"  # 4-byte pointers on wasm32: 20 M items is 80 MB
    fuel = 1_000_000_000  # filling 20 M slots costs more than the default per-call budget
    assert run_calls(src, "f", [[20_000_000]], mem_mb=256, fuel=fuel)[0].value == 20_000_000
    assert run_calls(src, "f", [[20_000_000]], mem_mb=64, fuel=fuel)[0].kind == "resource"


def test_c_stack_exhaustion_is_resource_and_ends_the_batch():
    src = ("def f(n):\n"
           "    v = []\n    for _ in range(n):\n        v = [v]\n"
           "    return len(str(v))\n")
    results = run_calls(src, "f", [[3], [100_000], [4]], fuel=500_000_000)
    assert [(r.kind, r.value) for r in results] == [("ok", 8), ("resource", None), ("skipped", None)]
    assert results[1].error == "wasm stack exhausted" and results[1].fuel > 0


def test_python_recursion_is_resource_not_an_exception():
    [r] = run_calls("def f(x):\n    return f(x)\n", "f", [[0]])
    assert (r.kind, r.value) == ("resource", None)


# Codex P2: a caught MemoryError/RecursionError used to come back as a settled ok, and the cap could
# then choose the value. A RAISE monitor in the guest flags the call however the candidate handles it.
CAUGHT = "def f(n):\n    try:\n        a = [0] * n\n    except Exception:\n        return 123\n    return 456\n"


@pytest.mark.parametrize("src, call", [
    (CAUGHT, [10**9]),
    ("def f(n):\n    try:\n        a = 'x' * n\n    except:\n        return 1\n", [10**9]),
    ("def f(n):\n    try:\n        return sorted([1, 2], key=lambda v: [0] * n)\n    except Exception:\n        return 9\n", [10**9]),
    ("def f(n):\n    out = []\n    for g in (lambda: [0] * n, lambda: list(range(n))):\n"
     "        try:\n            g()\n        except Exception as e:\n            out.append(1)\n    return out\n", [3 * 10**8]),
    ("def g():\n    return g()\ndef f(n):\n    try:\n        g()\n    except Exception:\n        return n\n", [5]),
])
def test_a_caught_resource_failure_is_still_resource(src, call):
    [r] = run_calls(src, "f", [call])
    assert r.kind == "resource", r


# Codex follow-up #1: the resource check ran before the call's objects were released, so a generator
# whose finally raised and caught RecursionError during cleanup left a settled exc behind.
CLEANUP_RECURSION = ("def f():\n"
                     "    def g():\n        try:\n            yield 1\n        finally:\n"
                     "            try:\n                raise RecursionError()\n            except Exception:\n                pass\n"
                     "    def started():\n        it = g(); next(it); return it\n"
                     "    raise ValueError(started())\n")
CLEANUP_CYCLE = ("def f():\n"
                 "    def g():\n        try:\n            yield 1\n        finally:\n"
                 "            try:\n                raise RecursionError()\n            except Exception:\n                pass\n"
                 "    box = []\n    it = g(); next(it); box.append(it); box.append(box)  # only a collection frees it\n"
                 "    return 1\n")


@pytest.mark.parametrize("src", [CLEANUP_RECURSION, CLEANUP_CYCLE])
def test_a_resource_failure_during_cleanup_is_still_resource(src):
    assert static_gate(src) == []
    assert kinds(run_calls(src, "f", [[], []])) == ["resource", "resource"]


def test_the_memory_cap_cannot_choose_a_settled_value():
    big = 10**9  # the fill costs more than the default budget
    small = run_calls(CAUGHT, "f", [[20_000_000]], mem_mb=64, fuel=big)
    large = run_calls(CAUGHT, "f", [[20_000_000]], mem_mb=256, fuel=big)
    assert (small[0].kind, large[0].kind, large[0].value) == ("resource", "ok", 456)


def test_resource_flag_is_per_call_and_ordinary_exceptions_are_unaffected():
    results = run_calls(CAUGHT, "f", [[5], [10**9], [7]])
    assert [(r.kind, r.value) for r in results] == [("ok", 456), ("resource", None), ("ok", 456)]
    src = "def f(x):\n    try:\n        return int(x)\n    except ValueError:\n        return -1\n"
    assert [r.value for r in run_calls(src, "f", [["a"], ["3"]])] == [-1, 3]


def test_a_caught_resource_failure_in_the_module_body_is_resource_for_every_call():
    src = "try:\n    X = [0] * 10**9\nexcept Exception:\n    X = 1\ndef f():\n    return X\n"
    assert kinds(run_calls(src, "f", [[], []])) == ["resource", "resource"]


def test_the_monitor_is_unreachable_from_candidate_code():
    assert static_gate("import sys\n") == ["import of 'sys' is not allowed"]
    assert static_gate("def f():\n    return '{0.monitoring}'.format(1)\n")
    [r] = run_calls("def f():\n    import sys\n    return 1\n", "f", [[]], _skip_gate=True)
    assert (r.kind, r.value) == ("exc", "ImportError")


def test_memory_error_name_is_banned_so_resource_cannot_be_faked():
    assert static_gate("def f():\n    raise MemoryError()\n") == ["banned name 'MemoryError'"]
    assert static_gate("def f():\n    try:\n        pass\n    except MemoryError:\n        pass\n")
    [r] = run_calls("def f():\n    raise MemoryError()\n", "f", [[]], _skip_gate=True)
    assert (r.kind, r.value) == ("exc", "NameError")  # not even reachable at runtime


# --------------------------------------------------------------------------- WASI surface (replaces rlimits)

def test_guest_sees_no_host():
    probe = sandbox._probe_guest("surface")
    assert probe["version"] == "3.14.7"
    assert probe["env"] == {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    assert probe["path"][0] == "/lib/python314.zip" and "" not in probe["path"]
    assert probe["lib"] == ["python314.zip"] and probe["root"] == "error: FileNotFoundError"
    assert probe["time"] == 1_700_000_000.0  # fixed wall clock
    assert probe["monotonic_ns"][1] - probe["monotonic_ns"][0] == 1000  # stepped, never real
    assert probe["sleep"].startswith("error")
    assert probe["write_lib"] == "PermissionError" and probe["listdir_up"] == "PermissionError"
    assert probe["read_etc"] == "FileNotFoundError" and probe["socket"] != "allowed"
    assert probe["urandom"] == sandbox._probe_guest("surface")["urandom"]  # seeded, reproducible
    assert probe["missing"] == {}  # every frozen SAFE_EXPORTS name exists in the guest
    # No host metadata: wasmtime's own preopen showed host inode numbers, link counts, sizes and real
    # timestamps (atime included), which moved fuel between a host and a container on one machine.
    fixed = 1_700_000_000
    assert probe["stat_lib"] == [0o040000, 2, 0, 1, 0, 0, 0, fixed, fixed, fixed]
    zip_size = (sandbox.runtime_root() / "lib" / "python314.zip").stat().st_size
    assert probe["stat_zip"] == [0o100000, 3, 0, 1, 0, 0, zip_size, fixed, fixed, fixed]
    assert probe["scandir_ino"] == [3]


def test_nothing_reachable_from_the_exports_is_a_module_frame_code_or_loader():
    probe = sandbox._probe_guest("reach")
    assert probe["found"] == [] and probe["visited"] > 1000


def test_typing_and_operator_cannot_reach_os_sys_or_the_loader():
    src = ("import typing, operator\n"
           "def f(x):\n"
           "    probes = [lambda: typing.sys, lambda: typing.functools, lambda: typing.collections,\n"
           "              lambda: typing.types, lambda: typing.copyreg, lambda: typing.operator,\n"
           "              lambda: typing.get_type_hints, lambda: typing.get_args, lambda: typing.ForwardRef,\n"
           "              lambda: typing.evaluate_forward_ref, lambda: typing.TypeVar, lambda: typing.NamedTuple,\n"
           "              lambda: typing.Protocol, lambda: operator.attrgetter, lambda: operator.methodcaller,\n"
           "              lambda: operator.call]\n"
           "    return probes[x]()\n")
    assert static_gate(src)  # the gate already refuses the helper names; prove the proxies lack them too
    results = run_calls(src, "f", [[i] for i in range(16)], _skip_gate=True)
    assert [(r.kind, r.value) for r in results] == [("exc", "AttributeError")] * 16
    loader = ("import typing, operator\n"
              "def f():\n"
              "    return [m.__loader__ is None and m.__spec__ is None for m in (typing, operator)] + [typing.__name__]\n")
    assert run_calls(loader, "f", [[]], _skip_gate=True)[0].value == [True, True, "_arena_safe_typing"]


@pytest.mark.parametrize("src", [
    "from typing import get_type_hints\n",
    "from typing import get_args\n",
    "from typing import ForwardRef\n",
    "from typing import sys\n",
    "from operator import attrgetter\n",
    "from operator import methodcaller\n",
    "import typing\ndef f(g):\n    return typing.get_type_hints(g)\n",
    "import operator\ndef f(x):\n    return operator.attrgetter('real')(x)\n",
    "def f(ref):\n    return ref.evaluate()\n",
    "def f(tv):\n    return tv.evaluate_bound\n",
])
def test_static_gate_rejects_string_evaluation_helpers(src):
    assert static_gate(src)


def test_string_annotations_are_never_evaluated():
    src = ("import typing\n"
           "def g(x: 'f.__globals__'):\n    return x\n"
           "def f(n):\n    return [g(n), typing.List['os.system'] is not None]\n")
    assert static_gate(src) == []  # an annotation string reaches no format path
    [r] = run_calls(src, "f", [[5]])  # and the runtime never evaluates it
    assert (r.kind, r.value) == ("ok", [5, True])


# --------------------------------------------------------------------------- speed and concurrency

BENCH_SRC = ("def f(xs, k):\n"
             "    out = []\n"
             "    for i, x in enumerate(xs):\n"
             "        if x % k == 0 and i < 10:\n"
             "            out.append(x * 2 + 1)\n"
             "    return sorted(out)\n")
BENCH_CALLS = [[list(range(i, i + 20)), (i % 5) + 1] for i in range(64)]


def test_batch_of_64_calls_costs_a_fixed_small_amount_of_fuel():
    # The gating cost check is in fuel, which load cannot move: wall-clock asserts flaked on ruos.
    first, second = run_calls(BENCH_SRC, "f", BENCH_CALLS), run_calls(BENCH_SRC, "f", BENCH_CALLS)
    assert kinds(first) == ["ok"] * 64 and outcome(first) == outcome(second)
    assert sum(r.fuel for r in first) == 39_940_964 and max(r.fuel for r in first) < FUEL_PER_CALL // 100


@pytest.mark.skipif(os.environ.get("QE_ARENA_BENCH") != "1", reason="wall-clock benchmark; set QE_ARENA_BENCH=1")
def test_batch_of_64_calls_is_fast():
    """Non-gating: wall time depends on the machine and its load. Run with QE_ARENA_BENCH=1."""
    timings = []
    for _ in range(5):
        t = time.perf_counter()
        results = run_calls(BENCH_SRC, "f", BENCH_CALLS)
        timings.append(time.perf_counter() - t)
        assert kinds(results) == ["ok"] * 64
    assert pystats.median(timings) < (0.3 if LINUX else 1.0), timings


def test_string_hashing_is_deterministic_across_batches():
    # Reference and candidate run in different instances; set iteration order must agree.
    src = "def f(words):\n    return [list(set(words)), hash(words[0]) % 1000003]\n"
    call = [["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]]
    runs = [run_calls(src, "f", [call])[0] for _ in range(4)]
    assert all(r.kind == "ok" for r in runs)
    assert all(compare(runs[0], r) for r in runs[1:]), [r.value for r in runs]


def test_concurrent_batches_from_threads_do_not_interfere():
    src = "def f(x):\n    return x * 3\n"
    out: dict[int, list] = {}

    def worker(i: int) -> None:
        out[i] = [r.value for r in run_calls(src, "f", [[i], [i + 1]])]
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == {i: [3 * i, 3 * i + 3] for i in range(8)}


def _cgroup(tmp_path, **files):
    for name, text in files.items():
        path = tmp_path / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


@pytest.mark.parametrize("files, expected", [
    ({"cpu.max": "200000 100000\n"}, 2),          # docker run --cpus=2 on cgroup v2
    ({"cpu.max": "150000 100000\n"}, 2),          # fractional quota rounds up
    ({"cpu.max": "50000 100000\n"}, 1),           # never below one worker
    ({"cpu.max": "1600000 100000\n"}, 4),         # never above MAX_WORKERS
    ({"cpu.max": "300000\n"}, 3),                 # period omitted: the kernel default 100000
    ({"cpu__cpu.cfs_quota_us": "100000\n", "cpu__cpu.cfs_period_us": "100000\n"}, 1),  # cgroup v1
])
def test_worker_count_follows_the_cgroup_quota(tmp_path, files, expected):
    assert sandbox.detect_workers(_cgroup(tmp_path, **files)) == expected


@pytest.mark.parametrize("files", [{"cpu.max": "max 100000\n"}, {}, {"cpu.max": "garbage\n"},
                                   {"cpu__cpu.cfs_quota_us": "-1\n", "cpu__cpu.cfs_period_us": "100000\n"}])
def test_unlimited_or_unreadable_cgroup_falls_back_to_visible_cpus(tmp_path, files, monkeypatch):
    monkeypatch.setattr(sandbox.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert sandbox.detect_workers(_cgroup(tmp_path, **files)) == 3
    monkeypatch.delattr(sandbox.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(sandbox.os, "cpu_count", lambda: 16)
    assert sandbox.detect_workers(_cgroup(tmp_path, **files)) == sandbox.MAX_WORKERS
    monkeypatch.setattr(sandbox.os, "cpu_count", lambda: None)
    assert sandbox.detect_workers(_cgroup(tmp_path, **files)) == 1


def test_worker_count_override(monkeypatch):
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "7")
    assert sandbox._pool_size() == 7  # an explicit override is not capped
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "0")
    assert sandbox._pool_size() == 1
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "lots")
    monkeypatch.setattr(sandbox, "_DETECTED", [3])
    assert sandbox._pool_size() == 3
    monkeypatch.delenv("QE_ARENA_WASI_WORKERS")
    assert sandbox._pool_size() == 3


def test_a_dead_worker_is_replaced_and_the_batch_retried():
    worker = sandbox._acquire(time.monotonic() + 60)
    worker.proc.kill()
    worker.proc.wait()
    sandbox._release(worker, True)  # returned as if healthy: the next request finds it dead
    assert [r.value for r in run_calls("def f(x):\n    return -x\n", "f", [[2]])] == [-2]


# Codex P4: a stopped worker blocked the request thread in a pipe write past every deadline, pool
# acquisition could wait forever, and a failing warm-up leaked capacity until the pool deadlocked.
BIG = ["x" * 1_000_000]  # larger than any pipe buffer, so the write itself has to wait


def test_a_stopped_worker_cannot_block_a_write_past_its_deadline():
    worker = sandbox._Worker()
    try:
        os.kill(worker.proc.pid, signal.SIGSTOP)
        t = time.monotonic()
        status = worker.request({"id": "x", "calls": [BIG]}, time.monotonic() + 0.3)
        assert status == "timeout" and time.monotonic() - t < 30.0  # deadline 0.3 s; it used to hang
    finally:
        worker.close()  # SIGKILL ends a stopped process too
    assert worker.proc.returncode is not None


def test_a_stopped_pool_worker_is_replaced_within_the_backstop(monkeypatch):
    # The request to a stopped worker must end at its own deadline (not hang the thread), the worker must be
    # killed, and run_calls' retry must run on another worker. The test sets each attempt's deadline itself:
    # with one module-wide 2 s backstop the retry also had to start a fresh worker inside those 2 s, which
    # takes 4-6 s on a loaded host, so the test failed under load (fix/pool-backstop-under-load).
    worker = sandbox._acquire(time.monotonic() + 60)
    os.kill(worker.proc.pid, signal.SIGSTOP)
    sandbox._release(worker, True)  # back in the pool, apparently healthy; acquired next (LIFO)
    real, attempts, backstops = sandbox._dispatch, [], iter([0.5, 300.0])

    def dispatch(payload):
        monkeypatch.setattr(sandbox, "BATCH_BACKSTOP_S", next(backstops))
        response = real(payload)
        attempts.append(response if isinstance(response, str) else "response")
        return response
    monkeypatch.setattr(sandbox, "_dispatch", dispatch)
    results = run_calls("def f(s):\n    return len(s)\n", "f", [BIG])
    assert attempts == ["timeout", "response"]       # a stopped worker can only time out, never answer
    assert [r.value for r in results] == [1_000_000]  # the retry ran on another worker
    assert worker.proc.poll() is not None             # and the stopped one was killed, not reused


class _FakeWorker:
    made = 0

    def __init__(self, *args):
        type(self).made += 1
        if type(self).made == 2:
            raise RuntimeError("simulated startup failure")

    def close(self):
        pass


def test_pool_acquisition_has_a_deadline(monkeypatch):
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "1")
    monkeypatch.setattr(sandbox, "_IDLE", [])
    monkeypatch.setattr(sandbox, "_LIVE", [1])  # the only worker is busy elsewhere
    monkeypatch.setattr(sandbox, "_WAITING", collections.deque())  # isolate every piece of pool state:
    monkeypatch.setattr(sandbox, "_INFLIGHT", collections.Counter())  # a leaked count blocks later tests
    monkeypatch.setattr(sandbox, "BATCH_BACKSTOP_S", 0.3)
    t = time.monotonic()
    assert sandbox._dispatch({"probe": "surface"}) == "pool timeout"
    assert time.monotonic() - t < 30.0  # deadline 0.3 s; it used to wait forever
    assert kinds(run_calls("def f():\n    return 1\n", "f", [[]])) == ["infra"]


def test_a_failing_warm_up_returns_the_workers_it_started(monkeypatch):
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "3")
    monkeypatch.setattr(sandbox, "_IDLE", [])
    monkeypatch.setattr(sandbox, "_LIVE", [0])
    monkeypatch.setattr(sandbox, "_Worker", _FakeWorker)
    monkeypatch.setattr(sandbox, "_WAITING", collections.deque())  # isolate every piece of pool state:
    monkeypatch.setattr(sandbox, "_INFLIGHT", collections.Counter())  # a leaked count blocks later tests
    _FakeWorker.made = 0
    with pytest.raises(RuntimeError):
        sandbox.warm()
    assert sandbox._LIVE[0] == len(sandbox._IDLE) == 1  # the first worker came back; nothing leaked
    first = sandbox._acquire(time.monotonic() + 1)      # neither the pool nor a retry is stuck
    again = sandbox._acquire(time.monotonic() + 1)
    assert sandbox._LIVE[0] == 2 and first is not again


# Codex follow-up #3: worker startup waited WORKER_STARTUP_S regardless of the dispatch deadline.
SLOW_START = "import time; time.sleep(30); print('{\"ready\": true}', flush=True)"


def test_worker_startup_never_outlives_the_dispatch_deadline(monkeypatch):
    import collections  # noqa: PLC0415
    real = sandbox.subprocess.Popen
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda cmd, **kw: real([sys.executable, "-c", SLOW_START], **kw))
    t = time.monotonic()
    with pytest.raises(RuntimeError, match="did not start"):
        sandbox._Worker(time.monotonic() + 0.5)
    assert time.monotonic() - t < 5.0
    monkeypatch.setattr(sandbox, "_IDLE", [])
    monkeypatch.setattr(sandbox, "_LIVE", [0])
    monkeypatch.setattr(sandbox, "_WAITING", collections.deque())
    monkeypatch.setattr(sandbox, "_INFLIGHT", collections.Counter())
    monkeypatch.setattr(sandbox, "BATCH_BACKSTOP_S", 0.5)
    t = time.monotonic()
    assert sandbox._dispatch({"probe": "surface"}).startswith("worker start failed")
    assert time.monotonic() - t < 5.0 and sandbox._LIVE[0] == 0


def test_wasmtime_private_fuel_surface_is_pinned():
    """The worker sets fuel from inside host calls through wasmtime._ffi, which is private API.
    If an upgrade changes it, this fails loudly instead of the sandbox silently mis-metering."""
    assert importlib.metadata.version("wasmtime") == "47.0.1"
    from wasmtime import _ffi
    assert callable(_ffi.wasmtime_context_set_fuel) and callable(_ffi.wasmtime_context_get_fuel)
    assert callable(getattr(wt.Caller, "_context", None))
    cfg = wt.Config()
    cfg.consume_fuel = True
    engine = wt.Engine(cfg)
    store = wt.Store(engine)
    store.set_fuel(10_000_000)
    seen = []

    def host(caller):
        wasi_worker.set_caller_fuel(caller, 5000)
        seen.append(wasi_worker.get_caller_fuel(caller))
    module = wt.Module(engine, '(module (import "env" "h" (func $h)) '
                               '(func (export "run") call $h (loop $l (br $l))))')
    linker = wt.Linker(engine)
    linker.define_func("env", "h", wt.FuncType([], []), host, access_caller=True)
    with pytest.raises(wt.Trap) as trap:
        linker.instantiate(store, module).exports(store)["run"](store)
    assert trap.value.trap_code == wt.TrapCode.OUT_OF_FUEL
    assert seen == [5000] and store.get_fuel() == 0
    assert ctypes.sizeof(ctypes.c_uint64) == 8


# --------------------------------------------------------------------------- static gate

@pytest.mark.parametrize("src", [
    "import os\ndef f():\n    return 1\n",
    "from os import system\ndef f():\n    return 1\n",
    "import os.path\n",
    "import collections.abc\n",
    "from collections import abc\n",
    "from functools import update_wrapper\n",
    "from functools import wraps\n",
    "from functools import singledispatch\n",
    "from string import Formatter\n",
    "from statistics import sys\n",
    "from math import *\n",
    "from . import corpus\n",
    "from __future__ import annotations\n",
    "def f():\n    return __import__('os')\n",
    "def f():\n    return open('/etc/passwd').read()\n",
    "def f():\n    return ().__class__.__bases__[0].__subclasses__()\n",
    "def f(x):\n    return getattr(x, '__globals__')\n",
    "def f():\n    return globals()\n",
    "def f():\n    return locals()\n",
    "def f(x):\n    return vars(x)\n",
    "def f():\n    return [c for c in ().__class__.__mro__]\n",
    "def f():\n    return exec('1')\n",
    "def f():\n    return eval('1')\n",
    "def f():\n    return compile('1', 'x', 'eval')\n",
    "def f():\n    breakpoint()\n",
    "def f(x):\n    setattr(x, 'a', 1)\n",
    "def f(x):\n    delattr(x, 'a')\n",
    "def f():\n    return input()\n",
    "def f():\n    help()\n",
    "def f(b):\n    return memoryview(b)\n",
    "def f():\n    exit()\n",
    "def f():\n    quit()\n",
    "class A:\n    pass\n",
    "x = 0\ndef f():\n    global x\n    x = 1\n",
    "def f():\n    x = 0\n    def g():\n        nonlocal x\n        x = 1\n",
    "def f():\n    return __builtins__\n",
    "def f():\n    return __name__\n",
    "def f(__x):\n    return 1\n",
    "def __f():\n    return 1\n",
    "def f():\n    g = (i for i in [1])\n    return g.gi_frame\n",
    "def f():\n    g = (i for i in [1])\n    return g.gi_code\n",
    "def f(fr):\n    return fr.f_back.f_globals\n",
    "def f(tb):\n    return tb.tb_frame\n",
    "def f(c):\n    return c.co_consts\n",
    "def f(e):\n    return e.with_traceback(None)\n",
    "def f(x):\n    return x._private\n",
    "def f():\n    return int.mro()\n",
    "def f(s):\n    return s.format_map({})\n",
    "async def f():\n    return 1\n",
    "def f(x):\n    match x:\n        case object(gi_frame=g):\n            return g\n",
    "type T = int\n",
    "def f[T](x):\n    return x\n",
    "def f(:\n",
    "x = 1\n" * 1300,
    "def f():\n    return " + "-" * 150 + "1\n",
])
def test_static_gate_rejects(src):
    assert static_gate(src), src
    assert kinds(run_calls(src, "f", [[]])) == ["rejected"]


def test_static_gate_reports_each_problem_once():
    problems = static_gate("import os\nimport os\ndef f():\n    return eval(eval('1'))\n")
    assert problems == ["import of 'os' is not allowed", "banned name 'eval'"]


@pytest.mark.parametrize("src", [
    "import math\ndef f(x):\n    return math.floor(x)\n",
    "from collections import Counter, defaultdict\nimport re as regex\ndef f(s):\n    return regex.sub('a', 'b', s)\n",
    "def f(xs):\n    key = lambda v: -v\n    return sorted(xs, key=key)\n",
    "def f():\n    while True:\n        break\n    print('done')\n    return '{0}'.format(1)\n",
    "def f(x):\n    try:\n        return 1 // x\n    except ZeroDivisionError as err:\n        raise ValueError('x') from err\n",
    "def f(n):\n    def g(k):\n        yield from range(k)\n    return list(g(n))\n",
    "import functools\n@functools.lru_cache(maxsize=None)\ndef f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\n",
    "def f(s):\n    return f'{s!r:>10}'\n",
])
def test_static_gate_accepts_ordinary_code(src):
    assert static_gate(src) == []


def test_static_gate_size_and_type_boundaries():
    assert static_gate("#" * (sandbox.MAX_SOURCE_BYTES - 1) + "\n") == []
    assert static_gate("#" * sandbox.MAX_SOURCE_BYTES + "\n")
    assert static_gate("# " + "é" * 3000 + "\n")  # bytes, not characters, are counted
    assert static_gate(b"def f(): pass")  # type: ignore[arg-type]
    assert static_gate("def f():\n    return 1\x00\n")


def test_static_gate_sees_through_unicode_normalization():
    # Identifiers are NFKC-normalized by the parser: fullwidth letters spell the same attribute.
    assert static_gate("def f(g):\n    return g.ｇｉ_frame\n")
    assert static_gate("def f():\n    return ＿＿import＿＿('os')\n")


@pytest.mark.parametrize("path", sorted(CORPUS.glob("[!_]*.py")), ids=lambda p: p.stem)
def test_corpus_files_pass_the_gate(path):
    assert static_gate(path.read_text()) == []


# --------------------------------------------------------------------------- traversal (Codex review P1)

NONCE_LEAK = ("def f():\n    try:\n        1 / 0\n    except Exception as e:\n"
              "        return '{0.__traceback__.tb_frame.f_back.f_locals[nonce]}'.format(e)\n")
CALLS_LEAK = ("def f():\n    try:\n        1 / 0\n    except Exception as e:\n"
              "        return '{0.__traceback__.tb_frame.f_back.f_locals[req][calls]}'.format(e)\n")
TYPING_LEAK = "import typing\ndef f():\n    return '{0.__class__.__init__.__globals__[sys].version}'.format(typing.List)\n"


@pytest.mark.parametrize("src", [NONCE_LEAK, CALLS_LEAK, TYPING_LEAK])
def test_format_traversal_reproductions_are_rejected(src):
    assert static_gate(src)
    assert kinds(run_calls(src, "f", [[]])) == ["rejected"]


@pytest.mark.parametrize("src", [
    "def f(g):\n    s = '{0.' + 'gi_' + 'frame}'\n    return s.format(g)\n",   # built at runtime
    "def f(g):\n    return str.format('{0}', g)\n",                           # unbound method
    "def f(x):\n    return '{0.real}'.format(x)\n",                           # attribute field
    "def f(x):\n    return '{0[0]}'.format(x)\n",                             # index field
    "def f(x, w):\n    return '{0:{1.real}}'.format(x, w)\n",                 # nested spec field
    "def f(x):\n    return '{'.format(x)\n",                                  # malformed
    "def f(x):\n    h = format\n    return h(x, '>3')\n",                    # format passed around
    "def f(x, s):\n    return format(x, s)\n",                                # non-literal spec
    "def f(x):\n    return '{0} tb_frame'.format(x)\n",                     # traversal names in a format string
    "def f(x):\n    return '{0}__{0}'.format(x)\n",
    "def f(x):\n    return format(x, '__')\n",
])
def test_static_gate_rejects_runtime_traversal_routes(src):
    assert static_gate(src), src


@pytest.mark.parametrize("literal, rejected", [
    ("prefix_f_back", False), ("f_back_suffix", False), ("x_f_back_y", False), ("lines_of_code", False),
    ("f_back", True), ("x.f_back", True), ("[f_back]", True), ("a f_locals b", True), ("{0.tb_frame}", True),
])
def test_traversal_names_match_as_whole_words_on_both_sides(literal, rejected):
    # Codex follow-up #2: "_" ended a word on the left but continued it on the right, so
    # "prefix_f_back" was refused while "f_back_suffix" passed. Checked where it matters: in a format string.
    receiver = literal.replace("{", "{{").replace("}", "}}") + " {0}" if "{0." not in literal else literal
    assert bool(static_gate(f"def f(d):\n    return {receiver!r}.format(d)\n")) is rejected


# Red team on 6ba1905: a correct fix whose docstring mentioned __doc__ scored 0, and x.split('__') was
# refused. Only strings that reach a format path are checked for traversal names now.
@pytest.mark.parametrize("src", [
    'def f(x):\n    """Returns x; see __doc__ and f_back."""\n    return x\n',
    '"""Module docstring naming __init__."""\ndef f(x):\n    return x\n',
    "def f(x):\n    return x.split('__')\n",
    "def f(d):\n    return [d.get('f_back'), d.get('__builtins__'), 'tb_frame', '{0.f_code}']\n",
    "def f(x):\n    return '%(__x)s' % {'__x': x}\n",
])
def test_literals_outside_format_paths_are_accepted_and_inert(src):
    assert static_gate(src) == []
    results = run_calls(src, "f", [["a__b"]] if "split" in src else [[{"f_back": 1}]] if "d.get" in src else [[1]])
    assert results[0].kind == "ok", results[0]


def test_ordinary_formatting_still_works():
    src = ("def f(x, w):\n"
           "    return ['{:.2f}'.format(x), '{0}-{1}'.format(w, w), '{name}!'.format(name=w), f'{x:>{w}}',\n"
           "            format(x, '>8.3f'), format(w), '%s/%d' % (x, w), '{0:{1}}'.format(x, w), f'{w!r}']\n")
    assert static_gate(src) == []
    [r] = run_calls(src, "f", [[3.14159, 6]])
    assert r.value == ["3.14", "6-6", "6!", "3.14159", "   3.142", "6", "3.14159/6", "3.14159", "6"]


def test_frames_above_candidate_code_hold_no_nonce_or_pending_inputs(monkeypatch):
    # Even with the gate switched off, the frames above a call and the sys module must not hold the
    # nonce, the request or inputs that have not been passed yet.
    nonce = "feedface" * 4
    real = sandbox.secrets.token_hex
    monkeypatch.setattr(sandbox.secrets, "token_hex", lambda n: nonce if n == 16 else real(n))
    src = ("import typing\n"
           "def f(x):\n"
           "    try:\n        1 / 0\n"
           "    except Exception as e:\n        frame = e.__traceback__.tb_frame\n"
           "    seen = []\n"
           "    while frame is not None:\n"
           "        seen.append(repr(frame.f_locals))\n        seen.append(repr(frame.f_globals))\n"
           "        frame = frame.f_back\n"
           "    sys = typing.List.__class__.__init__.__globals__['sys']\n"
           "    return [len(seen), ' '.join(seen), sys.argv, sys.orig_argv]\n")
    results = run_calls(src, "f", [[1], ["SENTINEL-PENDING-INPUT"]], _skip_gate=True)
    assert results[0].kind == "ok", results[0]
    entries, dump, argv, orig_argv = results[0].value
    assert entries == 2 * 4  # four frames inspected: candidate, run closure, launcher lambda, root
    assert nonce not in dump and "SENTINEL-PENDING-INPUT" not in dump and "max_total_bytes" not in dump
    assert argv == ["python"] and orig_argv == []


def test_root_frame_globals_are_cleared(monkeypatch):
    src = ("def f():\n    try:\n        1 / 0\n    except Exception as e:\n        frame = e.__traceback__.tb_frame\n"
           "    while frame.f_back is not None:\n        frame = frame.f_back\n"
           "    return sorted(frame.f_globals)\n")
    assert run_calls(src, "f", [[]], _skip_gate=True)[0].value == []


# --------------------------------------------------------------------------- runtime layer alone

def test_restricted_builtins_block_open_and_import_without_the_gate():
    src = ("def f(x):\n"
           "    if x == 0:\n        return open('/etc/passwd').read()\n"
           "    if x == 1:\n        return __import__('os').getcwd()\n"
           "    if x == 2:\n        import os\n        return os.getcwd()\n"
           "    if x == 3:\n        from collections import abc\n        return 1\n"
           "    if x == 4:\n        import collections.abc\n        return 1\n"
           "    if x == 5:\n        return eval('1')\n"
           "    if x == 6:\n        return getattr(x, 'real')\n"
           "    if x == 7:\n        return type(x)\n"
           "    if x == 8:\n        from math import floor\n        return floor(2.5)\n")
    results = run_calls(src, "f", [[i] for i in range(9)], _skip_gate=True)
    assert [(r.kind, r.value) for r in results] == [
        ("exc", "NameError"), ("exc", "ImportError"), ("exc", "ImportError"), ("exc", "ImportError"),
        ("exc", "ImportError"), ("exc", "NameError"), ("exc", "NameError"), ("exc", "NameError"), ("ok", 2)]


def test_class_definition_fails_at_runtime_without_the_gate():
    results = run_calls("class A:\n    pass\ndef f():\n    return 1\n", "f", [[]], _skip_gate=True)
    assert results[0].kind == "crash" and "NameError" in results[0].error


def test_proxy_modules_hide_reflection_helpers_and_submodules():
    src = ("import functools, statistics, string, fractions, re, collections\n"
           "def f(x):\n"
           "    probes = [lambda: functools.update_wrapper, lambda: functools.wraps,\n"
           "              lambda: functools.singledispatch, lambda: string.Formatter,\n"
           "              lambda: statistics.sys, lambda: fractions.sys, lambda: fractions.decimal,\n"
           "              lambda: re.functools, lambda: collections.abc, lambda: re.enum]\n"
           "    return probes[x]()\n")
    results = run_calls(src, "f", [[i] for i in range(10)])
    assert [(r.kind, r.value) for r in results] == [("exc", "AttributeError")] * 10


def test_update_wrapper_globals_merge_escape_is_closed():
    # Classic escape: merge a library function's module globals (sys, real builtins) into ours.
    src = ("import functools, statistics\n"
           "def w():\n    pass\n"
           "def f(x):\n"
           "    functools.update_wrapper(w, statistics.mean, assigned=(), updated=('_' + '_globals__',))\n"
           "    return sys.modules['os'].getcwd()\n")
    [r] = run_calls(src, "f", [[0]])  # the functools proxy has no update_wrapper
    assert (r.kind, r.value) == ("exc", "AttributeError")


def test_str_format_traversal_is_rejected():
    # Formerly an accepted residual risk ("only yields strings"); the Codex review showed it reads the
    # nonce, the batch inputs and module globals, so the gate now refuses it.
    src = ("def f(x):\n"
           "    def g():\n        yield 1\n"
           "    gen = g()\n"
           "    return '{0.gi_frame}'.format(gen)[:6]\n")
    assert kinds(run_calls(src, "f", [[0]])) == ["rejected"]


# --------------------------------------------------------------------------- result channel

def test_printed_fake_result_lines_cannot_change_results():
    fake = json.dumps([{"k": "ok", "v": 999}])
    src = ("def f(x):\n"
           f"    print('\\x01R deadbeef ' + {fake!r})\n"
           "    print('\\x01E deadbeef')\n"
           "    return x\n")
    assert [r.value for r in run_calls(src, "f", [[1], [2]])] == [1, 2]


def test_returned_text_with_newlines_and_markers_cannot_inject_rows():
    src = "def f(x):\n    return 'a\\n\\x01E ' + x + '\\n\\x01R ' + x + ' {\"k\": \"ok\", \"v\": 1}\\n'\n"
    [r] = run_calls(src, "f", [["nonce"]])
    assert r.kind == "ok" and r.value == 'a\n\x01E nonce\n\x01R nonce {"k": "ok", "v": 1}\n'


def _protocol(nonce="n0", budgets=(500, 100)):
    fuel = {"left": 0, "sets": []}

    def set_fuel(amount):
        fuel["left"] = amount
        fuel["sets"].append(amount)
    proto = wasi_worker.Protocol(nonce, list(budgets), lambda: fuel["left"], set_fuel)
    return proto, fuel


def test_protocol_meters_calls_between_nonce_markers():
    proto, fuel = _protocol()
    proto.feed(b"\x01B n0\n")
    fuel["left"] -= 120
    proto.feed(b"\x01E n0\n\x01R n0 {\"k\": \"ok\"}\n\x01B n0\n")
    fuel["left"] -= 7
    proto.feed(b"\x01E n0\n\x01R n0 {\"k\": \"ok\", \"v\": 3}\n")
    assert proto.used == [120, 7] and proto.rows == [{"k": "ok"}, {"k": "ok", "v": 3}]
    assert fuel["sets"] == [500, wasi_worker.HARNESS_FUEL, 100, wasi_worker.HARNESS_FUEL]


@pytest.mark.parametrize("stream", [
    b"\x01B other\n",                               # wrong nonce
    b"\x01E n0\n",                                  # end before begin
    b"\x01B n0\n\x01B n0\n",                         # nested begin
    b"\x01B n0\n\x01R n0 {}\n",                      # row inside a call
    b"\x01B n0\n\x01E n0\n\x01B n0\n",                # begin before the previous row
    b"\x01B n0\n\x01E n0\n\x01R n0 [1]\n",            # row is not an object
    b"\x01B n0\n\x01E n0\n\x01R n0 {bad\n",           # row is not JSON
    b"hello\n",                                      # anything else on fd 1
    b"\x01B n0\n\x01E n0\n\x01R n0 {}\n\x01B n0\n\x01E n0\n\x01R n0 {}\n\x01B n0\n",  # more calls than budgets
])
def test_protocol_rejects_forged_or_out_of_order_markers(stream):
    proto, _ = _protocol()
    with pytest.raises(wasi_worker.ProtocolError):
        proto.feed(stream)


def test_protocol_caps_guest_output():
    proto, _ = _protocol()
    with pytest.raises(wasi_worker.ProtocolError):
        proto.feed(b"x" * (wasi_worker.MAX_GUEST_STDOUT + 1))


@pytest.mark.parametrize("data", [
    {"results": [{"k": "ok", "v": 1}]}, {"results": [{"k": "ok", "v": 1}, {"k": "weird"}]},
    {"results": [{"k": "ok", "v": 1}, {"k": "exc", "v": 3}]}, {"results": [{"k": "ok", "v": 1}, {"k": "ok", "x": 1}]},
    {"results": [{"k": "ok", "v": 1}, {"k": "rejected"}]}, {"results": [{"k": "ok", "v": 1}, "ok"]},
    {"results": [{"k": "ok", "v": 1}, {"k": "ok", "f": -1}]}, {"results": [{"k": "ok", "v": 1}, {"k": "ok", "f": 1.5}]},
    {"results": [{"k": "ok", "v": 1}, {"k": "timeout"}]}, {"error": "boom"}, "exited", None,
])
def test_parent_rejects_malformed_worker_responses(data):
    with pytest.raises(ValueError):
        sandbox._parse_results(data, 2)


def test_response_overflow_is_infra_not_a_result(monkeypatch):
    monkeypatch.setattr(sandbox, "MAX_RESPONSE_BYTES", 200)
    results = run_calls("def f(x):\n    return 'q' * 300\n", "f", [[0], [1]])
    assert kinds(results) == ["infra", "infra"]


def test_each_batch_uses_a_fresh_nonce(monkeypatch):
    seen = []
    real = sandbox.secrets.token_hex
    monkeypatch.setattr(sandbox.secrets, "token_hex", lambda n: seen.append(real(n)) or seen[-1])
    run_calls("def f():\n    return 1\n", "f", [[]])
    run_calls("def f():\n    return 1\n", "f", [[]])
    nonces = [s for s in seen if len(s) == 32]
    assert len(nonces) == 2 and nonces[0] != nonces[1]


# --------------------------------------------------------------------------- virtual filesystem (unit)

class _Memory:
    def __init__(self, size=4096):
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


def _fs():
    writes = []
    fs = wasi_fs.VirtualFS({"/lib/python314.zip": b"0123456789"}, lambda c, fd, data: writes.append((fd, data)) or 0)
    return fs, _Caller(), writes


def _put(caller, ptr, data):
    caller.memory.write(caller, data, ptr)
    return len(data)


def _open(fs, caller, path, oflags=0, rights=0, fdflags=0):
    n = _put(caller, 100, path.encode())
    errno = fs.path_open(caller, 3, 0, 100, n, oflags, rights, 0, fdflags, 200)
    return errno, int.from_bytes(caller.memory.data[200:204], "little")


def test_virtual_fs_exposes_one_read_only_preopen():
    fs, caller, _ = _fs()
    assert fs.fd_prestat_get(caller, 3, 0) == 0 and caller.memory.data[4:8] == (4).to_bytes(4, "little")
    assert fs.fd_prestat_dir_name(caller, 3, 16, 4) == 0 and bytes(caller.memory.data[16:20]) == b"/lib"
    assert fs.fd_prestat_get(caller, 4, 0) == wasi_fs.EBADF  # no second preopen
    for path in ("../etc/passwd", "/etc/passwd", "a/../../x"):
        assert _open(fs, caller, path)[0] == wasi_fs.EPERM, path
    assert _open(fs, caller, "missing.py")[0] == wasi_fs.ENOENT
    assert _open(fs, caller, "python314.zip", oflags=wasi_fs.O_CREAT)[0] == wasi_fs.EACCES
    assert _open(fs, caller, "python314.zip", oflags=wasi_fs.O_TRUNC)[0] == wasi_fs.EACCES
    assert _open(fs, caller, "python314.zip", rights=wasi_fs.RIGHT_FD_WRITE)[0] == wasi_fs.EACCES
    assert _open(fs, caller, "python314.zip", fdflags=wasi_fs.FDFLAG_APPEND)[0] == wasi_fs.EACCES
    assert _open(fs, caller, "python314.zip", oflags=wasi_fs.O_DIRECTORY)[0] == wasi_fs.ENOTDIR
    for refuse in (fs.path_create_directory, fs.path_remove_directory, fs.path_unlink_file):
        assert refuse(caller, 3, 100, 1) == wasi_fs.EROFS
    assert fs.fd_pwrite(caller, 3, 0, 0, 0, 0) == wasi_fs.EBADF


def test_virtual_fs_reads_seeks_and_preads():
    fs, caller, _ = _fs()
    errno, fd = _open(fs, caller, "python314.zip")
    assert errno == 0 and fd == 4
    iov = (300).to_bytes(4, "little") + (4).to_bytes(4, "little")
    _put(caller, 400, iov)
    assert fs.fd_read(caller, fd, 400, 1, 500) == 0 and bytes(caller.memory.data[300:304]) == b"0123"
    assert fs.fd_seek(caller, fd, -2, wasi_fs.WHENCE_END, 600) == 0
    assert fs.fd_read(caller, fd, 400, 1, 500) == 0 and bytes(caller.memory.data[300:302]) == b"89"
    assert int.from_bytes(caller.memory.data[500:504], "little") == 2
    assert fs.fd_pread(caller, fd, 400, 1, 3, 500) == 0 and bytes(caller.memory.data[300:304]) == b"3456"
    assert fs.fd_tell(caller, fd, 600) == 0 and int.from_bytes(caller.memory.data[600:608], "little") == 10
    assert fs.fd_seek(caller, fd, -1, wasi_fs.WHENCE_SET, 600) == wasi_fs.EINVAL
    assert fs.fd_close(caller, fd) == 0 and fs.fd_close(caller, fd) == wasi_fs.EBADF
    assert fs.fd_read(caller, 0, 400, 1, 500) == 0 and int.from_bytes(caller.memory.data[500:504], "little") == 0
    assert fs.fd_seek(caller, 1, 0, 0, 600) == wasi_fs.ESPIPE


def test_virtual_fs_metadata_is_canonical():
    fs, caller, _ = _fs()
    n = _put(caller, 100, b"python314.zip")
    assert fs.path_filestat_get(caller, 3, 0, 100, n, 800) == 0
    dev, ino, ftype, nlink, size, atim, mtim, ctim = __import__("struct").unpack_from("<QQB7xQQQQQ", caller.memory.data, 800)
    assert (dev, ftype, nlink, size) == (0, wasi_fs.REGULAR_FILE, 1, 10)
    assert atim == mtim == ctim == wasi_fs.FIXED_TIME_NS and ino == fs.inodes["/lib/python314.zip"]
    assert fs.fd_filestat_get(caller, 1, 800) == 0 and bytes(caller.memory.data[800:864]) == bytes(64)
    assert fs.fd_fdstat_get(caller, 0, 900) == 0 and bytes(caller.memory.data[900:924]) == bytes(24)  # not a tty


def test_virtual_fs_readdir_resumes_from_its_cookie_and_honours_the_buffer():
    fs, caller, _ = _fs()
    assert fs.fd_readdir(caller, 3, 1000, 2000, 0, 1500) == 0
    used = int.from_bytes(caller.memory.data[1500:1504], "little")
    raw, names, off = bytes(caller.memory.data[1000:1000 + used]), [], 0
    while off < len(raw):
        nxt, ino, namlen, dtype = __import__("struct").unpack_from("<QQIB", raw, off)
        names.append((raw[off + 24:off + 24 + namlen].decode(), nxt))
        off += 24 + namlen
    assert names == [(".", 1), ("..", 2), ("python314.zip", 3)]
    assert fs.fd_readdir(caller, 3, 1000, 30, 0, 1500) == 0  # truncated: the caller sees a full buffer
    assert int.from_bytes(caller.memory.data[1500:1504], "little") == 30
    assert fs.fd_readdir(caller, 3, 1000, 2000, 2, 1500) == 0  # resume after ".."
    assert caller.memory.data[1024:1024 + 13] == b"python314.zip"


def test_virtual_fs_routes_stdout_and_stderr_only():
    fs, caller, writes = _fs()
    _put(caller, 300, b"hi")
    _put(caller, 400, (300).to_bytes(4, "little") + (2).to_bytes(4, "little"))
    assert fs.fd_write(caller, 1, 400, 1, 500) == 0 and fs.fd_write(caller, 2, 400, 1, 500) == 0
    assert fs.fd_write(caller, 3, 400, 1, 500) == wasi_fs.EBADF
    assert writes == [(1, b"hi"), (2, b"hi")]


def test_virtual_fs_covers_every_file_import_of_python_wasm():
    module = wt.Module.from_file(wt.Engine(), str(sandbox.runtime_root() / "python.wasm"))
    imported = {i.name for i in module.imports if i.name.startswith(("fd_", "path_"))}
    assert imported == set(wasi_fs.SIGNATURES)
    for item in module.imports:
        if item.name in wasi_fs.SIGNATURES:
            params, results = wasi_fs.SIGNATURES[item.name]
            assert [str(t) for t in item.type.params] == params and [str(t) for t in item.type.results] == results


# --------------------------------------------------------------------------- compare

@pytest.mark.parametrize("a, b, equal", [
    (ok(True), ok(1), False),
    (ok(False), ok(0), False),
    (ok([True]), ok([1]), False),
    (ok(1), ok(1.0), False),
    (ok(1), ok(1), True),
    (ok(0.1 + 0.2), ok(0.3), True),
    (ok(1e-13), ok(0.0), True),
    (ok(1.0), ok(1.0 + 1e-6), False),
    (ok(1e20), ok(1e20 * (1 + 1e-10)), True),
    (ok(float("nan")), ok(float("nan")), True),
    (ok(float("inf")), ok(float("inf")), True),
    (ok(float("inf")), ok(float("-inf")), False),
    (ok(float("nan")), ok(0.0), False),
    (ok((1, (2, 3))), ok([1, [2, 3]]), True),
    (ok({"a": [1.0, None]}), ok({"a": [1.0 + 1e-12, None]}), True),
    (ok({"a": 1}), ok({"a": 1, "b": 2}), False),
    (ok([1, 2]), ok([1, 2, 3]), False),
    (ok("1"), ok(1), False),
    (ok(None), ok(None), True),
    (ok(None), ok(0), False),
    (CallResult("exc", "ValueError"), CallResult("exc", "ValueError"), True),
    (CallResult("exc", "ValueError"), CallResult("exc", "TypeError"), False),
    (CallResult("exc", "ValueError"), ok("ValueError"), False),
    (ok("ValueError"), CallResult("exc", "ValueError"), False),
    (CallResult("timeout"), CallResult("timeout"), False),
    (CallResult("fuel", fuel=5), CallResult("fuel", fuel=5), False),
    (CallResult("resource"), CallResult("resource"), False),
    (CallResult("infra"), CallResult("infra"), False),
    (CallResult("ok", 3, fuel=10), CallResult("ok", 3, fuel=99), True),
    (CallResult("crash"), CallResult("crash"), False),
    (CallResult("rejected"), CallResult("rejected"), False),
    (ok(1), CallResult("timeout"), False),
])
def test_compare(a, b, equal):
    assert compare(a, b) is equal
    assert compare(b, a) is equal


def test_compare_over_real_sandbox_outputs():
    src = "def f(x):\n    return [x > 0, x * 1.5, (x, str(x))]\n"
    [a] = run_calls(src, "f", [[2]])
    [b] = run_calls(src.replace("x > 0", "int(x > 0)"), "f", [[2]])
    assert compare(a, a) and not compare(a, b)
