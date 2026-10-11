"""Sandbox and oracle comparison. This is the reward function: tests think like an attacker."""

from __future__ import annotations

import json
import pathlib
import statistics as pystats
import sys
import threading
import time

import pytest

from qe_arena import sandbox
from qe_arena.sandbox import CallResult, compare, run_calls, static_gate

LINUX = sys.platform.startswith("linux")
CORPUS = pathlib.Path(__file__).resolve().parent.parent / "qe_arena" / "corpus"


def ok(value):
    return CallResult("ok", value)


def kinds(results):
    return [r.kind for r in results]


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
        ("exc", "ValueError"), ("exc", "KeyError"), ("exc", "ZeroDivisionError"), ("exc", "RecursionError"),
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
    assert [(r.kind, r.error) for r in results] == [("crash", "module: ZeroDivisionError: integer division or modulo by zero")] * 2


def test_print_is_a_silent_noop():
    src = "def f(x):\n    print('hello', x, sep='-')\n    return x\n"
    assert [r.value for r in run_calls(src, "f", [[1], [2]])] == [1, 2]


def test_allowlisted_modules_work():
    src = ('import math, re, itertools, functools, collections, string, bisect, heapq, statistics\n'
           'from decimal import Decimal\nfrom fractions import Fraction\n'
           'def f(xs):\n'
           '    c = collections.Counter(xs)\n'
           '    return [math.isqrt(sum(xs)), re.findall(r"\\d", "a1b2"), list(itertools.accumulate(xs)),\n'
           '            functools.reduce(lambda a, b: a * b, xs, 1), c.most_common(1)[0][0], string.digits[:3],\n'
           '            bisect.bisect_left(sorted(xs), 3), heapq.nsmallest(2, xs), statistics.median(xs),\n'
           '            str(Decimal("1.10") + Decimal("2.205")), str(Fraction(1, 3) + Fraction(1, 6))]\n')
    [r] = run_calls(src, "f", [[[4, 1, 3, 3, 9]]])
    assert r.kind == "ok", r
    assert r.value == [4, ["1", "2"], [4, 5, 8, 11, 20], 324, 3, "012", 1, [1, 3], 3, "3.305", "1/2"]


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


# --------------------------------------------------------------------------- time and memory

def test_per_call_timeout_spares_other_calls():
    src = "def f(x):\n    while x == 1:\n        pass\n    return x\n"
    t = time.monotonic()
    results = run_calls(src, "f", [[0], [1], [2]], per_call_timeout_s=0.3)
    assert [(r.kind, r.value) for r in results] == [("ok", 0), ("timeout", None), ("ok", 2)]
    assert time.monotonic() - t < 2.0


def test_swallowed_timeout_still_counts_as_timeout():
    src = ("def f(x):\n"
           "    try:\n"
           "        while True:\n            pass\n"
           "    except:\n"
           "        return 42\n")
    assert kinds(run_calls(src, "f", [[0], [1]], per_call_timeout_s=0.2)) == ["timeout", "timeout"]


def test_timeout_swallowed_forever_hits_the_batch_limit():
    src = ("def f(x):\n"
           "    while True:\n"
           "        try:\n            while True:\n                pass\n"
           "        except:\n            pass\n")
    t = time.monotonic()
    results = run_calls(src, "f", [[0], [1]], timeout_s=1.5, per_call_timeout_s=0.2)
    assert kinds(results) == ["timeout", "timeout"]
    assert time.monotonic() - t < 1.5 + sandbox.KILL_GRACE_S + 1.0


def test_c_level_loop_ignoring_signals_hits_the_batch_limit():
    src = "def f(x):\n    return sum(range(10 ** 13))\n"
    t = time.monotonic()
    results = run_calls(src, "f", [[0], [1]], timeout_s=1.0, per_call_timeout_s=0.2)
    assert kinds(results) == ["timeout", "timeout"]
    assert time.monotonic() - t < 1.0 + sandbox.KILL_GRACE_S + 1.0


def test_batch_budget_marks_unstarted_calls_as_timeout():
    src = "def f(x):\n    while True:\n        pass\n"
    results = run_calls(src, "f", [[i] for i in range(20)], timeout_s=1.0, per_call_timeout_s=0.3)
    assert kinds(results) == ["timeout"] * 20
    assert any(r.error == "batch budget exhausted" for r in results)


def test_module_level_infinite_loop_times_out():
    t = time.monotonic()
    results = run_calls("while True:\n    pass\ndef f():\n    return 1\n", "f", [[], []], timeout_s=1.5,
                        per_call_timeout_s=0.2)
    assert kinds(results) == ["timeout", "timeout"]
    assert time.monotonic() - t < 2.5


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is only enforceable on Linux; macOS refuses it")
def test_memory_bomb_fails_fast_on_linux():
    t = time.monotonic()
    results = run_calls("def f(x):\n    return len([0] * 10 ** 10)\n", "f", [[0], [1]])
    assert all((r.kind == "exc" and r.value == "MemoryError") or r.kind == "crash" for r in results), results
    assert time.monotonic() - t < 2.0


@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is only enforceable on Linux; macOS refuses it")
def test_gradual_memory_growth_is_capped_on_linux():
    src = "def f(x):\n    acc = []\n    while True:\n        acc.append('z' * 1_000_000)\n"
    results = run_calls(src, "f", [[0], [1]], per_call_timeout_s=2.0)
    assert [(r.kind, r.value) for r in results] == [("exc", "MemoryError")] * 2


@pytest.mark.skipif(not LINUX, reason="rlimits are asserted exactly on Linux")
def test_child_rlimits_are_applied_on_linux():
    probe = sandbox._probe_limits(mem_mb=256, timeout_s=3.0)
    assert probe["applied"] == {
        "RLIMIT_AS": [256 << 20, 256 << 20], "RLIMIT_CPU": [4, 5], "RLIMIT_FSIZE": [0, 0],
        "RLIMIT_NOFILE": [16, 16], "RLIMIT_CORE": [0, 0]}


def test_child_environment_is_empty_isolated_and_degrades_gracefully():
    probe = sandbox._probe_limits()
    assert probe["flags"] == {"no_site": 1, "no_user_site": 1, "safe_path": True, "hash_randomization": 0}
    assert probe["cwd_entries"] == 0
    assert "" not in probe["path"] and not any("site-packages" in p for p in probe["path"])
    # Python itself may set LC_CTYPE (PEP 538); macOS CoreFoundation adds its encoding key.
    assert set(probe["env"]) <= {"PYTHONHASHSEED", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}
    applied = probe["applied"]
    assert applied["RLIMIT_FSIZE"] == [0, 0] and applied["RLIMIT_CORE"] == [0, 0]
    assert applied["RLIMIT_NOFILE"] == [16, 16] and applied["RLIMIT_CPU"] == [4, 5]
    assert applied["RLIMIT_AS"] == [256 << 20] * 2 or str(applied["RLIMIT_AS"]).startswith("failed")


def test_batch_of_64_calls_is_fast():
    src = ("def f(xs, k):\n"
           "    out = []\n"
           "    for i, x in enumerate(xs):\n"
           "        if x % k == 0 and i < 10:\n"
           "            out.append(x * 2 + 1)\n"
           "    return sorted(out)\n")
    calls = [[list(range(i, i + 20)), (i % 5) + 1] for i in range(64)]
    timings = []
    for _ in range(5):
        t = time.perf_counter()
        results = run_calls(src, "f", calls)
        timings.append(time.perf_counter() - t)
        assert kinds(results) == ["ok"] * 64
    assert pystats.median(timings) < (0.3 if LINUX else 1.0), timings


def test_string_hashing_is_deterministic_across_batches():
    # Reference and candidate run in different processes; set iteration order must agree.
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
    [r] = run_calls(src, "f", [[0]])
    assert (r.kind, r.value) == ("exc", "AttributeError")


def test_str_format_traversal_yields_only_strings():
    # str.format can walk attributes at runtime; it can only ever produce text.
    src = ("def f(x):\n"
           "    def g():\n        yield 1\n"
           "    gen = g()\n"
           "    return '{0.gi_frame}'.format(gen)[:6]\n")
    [r] = run_calls(src, "f", [[0]])
    assert r.kind == "ok" and isinstance(r.value, str)


# --------------------------------------------------------------------------- result channel

def test_printed_fake_result_lines_cannot_change_results():
    fake = json.dumps([{"k": "ok", "v": 999}])
    src = ("def f(x):\n"
           f"    print('deadbeef ' + {fake!r})\n"
           f"    print({fake!r})\n"
           "    return x\n")
    assert [r.value for r in run_calls(src, "f", [[1], [2]])] == [1, 2]


def test_returned_text_with_newlines_cannot_inject_lines():
    src = "def f(x):\n    return 'a\\n' + x + ' [{\"k\": \"ok\", \"v\": 1}]\\n'\n"
    [r] = run_calls(src, "f", [["nonce"]])
    assert r.kind == "ok" and r.value == 'a\nnonce [{"k": "ok", "v": 1}]\n'


def test_parent_accepts_exactly_one_nonce_line():
    good = json.dumps([{"k": "ok", "v": 1}])
    assert sandbox._result_line(f"noise\nabc {good}\nmore\n".encode(), "abc") == [{"k": "ok", "v": 1}]
    with pytest.raises(ValueError):
        sandbox._result_line(f"abc {good}\nabc {good}\n".encode(), "abc")
    with pytest.raises(ValueError):
        sandbox._result_line(f"xyz {good}\n".encode(), "abc")


@pytest.mark.parametrize("data", [
    {"k": "ok"}, [{"k": "ok", "v": 1}], [{"k": "ok", "v": 1}, {"k": "weird"}],
    [{"k": "ok", "v": 1}, {"k": "exc", "v": 3}], [{"k": "ok", "v": 1}, {"k": "ok", "x": 1}],
    [{"k": "ok", "v": 1}, {"k": "rejected"}], [{"k": "ok", "v": 1}, "ok"],
])
def test_parent_rejects_malformed_result_payloads(data):
    with pytest.raises(ValueError):
        sandbox._parse_results(data, 2)


def test_stdout_overflow_is_a_crash(monkeypatch):
    monkeypatch.setattr(sandbox, "MAX_STDOUT_BYTES", 200)
    results = run_calls("def f(x):\n    return 'q' * 300\n", "f", [[0], [1]])
    assert [(r.kind, r.error) for r in results] == [("crash", "sandbox: overflow")] * 2


def test_each_batch_uses_a_fresh_nonce(monkeypatch):
    seen = []
    real = sandbox.secrets.token_hex
    monkeypatch.setattr(sandbox.secrets, "token_hex", lambda n: seen.append(real(n)) or seen[-1])
    run_calls("def f():\n    return 1\n", "f", [[]])
    run_calls("def f():\n    return 1\n", "f", [[]])
    assert len(seen) == 2 and seen[0] != seen[1] and len(seen[0]) == 32


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
