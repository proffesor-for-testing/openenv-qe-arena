"""Pinned gate limits, re.DEBUG flag positions, format-problem reporting, cgroup edges, send() and pool
edges. Added after the focused mutation pass on 6ba1905: kills 53 of the 54 non-equivalent survivors."""

from __future__ import annotations

import ast

import pytest

from qe_arena import sandbox


def test_gate_limits_are_pinned_literals():
    # Kills MAX_SOURCE_BYTES / MAX_AST_DEPTH off-by-ones: the gate tests read both back from the module.
    assert sandbox.MAX_SOURCE_BYTES == 6000 and sandbox.MAX_AST_DEPTH == 100
    assert sandbox.static_gate("#" * 5999 + "\n") == []
    assert sandbox.static_gate("#" * 6000 + "\n")


def test_ast_depth_counts_the_module_as_one():
    # Kills E-sandbox-193 (start depth / initial deepest): the depth tests derive k from _ast_depth itself.
    assert sandbox._ast_depth(ast.parse("")) == 1
    assert sandbox._ast_depth(ast.parse("x")) == 4          # Module > Expr > Name > Load
    deep = "def f():\n    return " + "-" * 96 + "1\n"       # depth exactly 100: allowed
    assert sandbox._ast_depth(ast.parse(deep)) == 100 and sandbox.static_gate(deep) == []
    deeper = "def f():\n    return " + "-" * 97 + "1\n"
    assert sandbox._ast_depth(ast.parse(deeper)) == 101 and sandbox.static_gate(deeper)


# Positional arguments before the flags parameter, from the re module's signatures.
RE_ARGS_BEFORE_FLAGS = {"compile": ["'a'"], "search": ["'a'", "s"], "match": ["'a'", "s"], "fullmatch": ["'a'", "s"],
                        "findall": ["'a'", "s"], "finditer": ["'a'", "s"], "split": ["'a'", "s", "0"],
                        "sub": ["'a'", "'b'", "s", "0"], "subn": ["'a'", "'b'", "s", "0"]}


def _re_src(call):
    return f"import re\ndef f(s):\n    return re.{call}\n"


@pytest.mark.parametrize("name", sorted(RE_ARGS_BEFORE_FLAGS))
def test_re_debug_is_refused_exactly_where_flags_go(name):
    # Kills the RE_FLAGS_POSITION and RE_DEBUG off-by-ones and the flags= keyword check (cmp_negate).
    assert sandbox.RE_FLAGS_POSITION[name] == len(RE_ARGS_BEFORE_FLAGS[name])
    args = RE_ARGS_BEFORE_FLAGS[name]
    debug = ["re.DEBUG is not allowed"]
    assert sandbox.static_gate(_re_src(f"{name}({', '.join(args + ['128'])})")) == debug
    assert sandbox.static_gate(_re_src(f"{name}({', '.join(args)}, flags=2 | 128)")) == debug
    assert sandbox.static_gate(_re_src(f"{name}({', '.join(args + ['3'])})")) == []        # IGNORECASE|LOCALE bits
    before = args[:-1] + ["128"]                                                          # 128 where flags is not
    assert sandbox.static_gate(_re_src(f"{name}({', '.join(before)})")) == []


def test_persistent_infra_raises_infra_error_with_a_bounded_message(monkeypatch):
    # Kills E-tasks-112 (message truncation) on top of the existing InfraError tests.
    from qe_arena import tasks
    calls = []

    def broken(source, entry, batch, **kw):
        calls.append(1)
        return [sandbox.CallResult("infra", error="x" * 1000) for _ in batch]
    monkeypatch.setattr(tasks.sandbox, "run_calls", broken)
    with pytest.raises(sandbox.InfraError) as info:
        tasks._outputs("def f():\n    return 1\n", "f", [[]])
    assert len(calls) == 2 and len(str(info.value)) == 300


def test_gate_reports_every_formatting_problem_not_just_the_first():
    # Kills E-sandbox-249/254 (continue -> break): one bad .format must not hide the next problem.
    src = ("def f(x, t):\n"
           "    a = t.format(1)\n"            # non-literal receiver
           "    b = '{0[k]}'.format(x)\n"     # index lookup
           "    c = '{'.format(x)\n"          # malformed
           "    return '{0} f_back'.format(x)\n")  # traversal word in a format string (39e39c1: only there)
    problems = sandbox.static_gate(src)
    for expected in (".format needs a string literal receiver", "format fields may not use attribute or index lookups",
                     "malformed format string", "format string mentions 'f_back'"):
        assert expected in problems, problems


def test_an_empty_cpu_max_file_falls_back_to_visible_cpus(tmp_path, monkeypatch):
    # Kills E-sandbox-441 (and -> or): an empty cpu.max raised IndexError out of detect_workers.
    (tmp_path / "cpu.max").write_text("")
    monkeypatch.setattr(sandbox.os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    assert sandbox.detect_workers(tmp_path) == 2


def _sender():
    import os
    import types
    r, w = os.pipe()
    os.set_blocking(w, False)
    worker = object.__new__(sandbox._Worker)
    worker.proc = types.SimpleNamespace(stdin=os.fdopen(w, "wb", buffering=0))
    return worker, r


def test_send_times_out_on_a_worker_that_stops_reading_and_never_claims_success():
    # Kills E-sandbox-391 (deadline arithmetic) and E-sandbox-394 (continue -> break claimed "sent").
    import os
    import time
    worker, r = _sender()
    t = time.monotonic()
    assert worker.send(b"x" * (4 << 20), time.monotonic() + 0.3) == "timeout"   # nobody reads: pipe fills
    assert time.monotonic() - t < 0.9
    os.close(r)
    assert worker.send(b"y" * 10, time.monotonic() + 1) == "exited"             # reader gone


def test_send_delivers_everything_through_a_slow_reader(monkeypatch):
    # Kills E-sandbox-398 (BlockingIOError -> break): a spurious EAGAIN must be retried, not end the send.
    import os
    import threading
    worker, r = _sender()
    real_write, state = os.write, {"n": 0}

    def flaky_write(fd, data):
        state["n"] += 1
        if state["n"] == 1:
            raise BlockingIOError
        return real_write(fd, data)
    monkeypatch.setattr(sandbox.os, "write", flaky_write)
    got = bytearray()
    reader = threading.Thread(target=lambda: [got.extend(c) for c in iter(lambda: os.read(r, 65536), b"")])
    reader.start()
    payload = b"z" * 300_000
    import time
    assert worker.send(payload, time.monotonic() + 10) == "sent"
    worker.proc.stdin.close()
    reader.join(10)
    assert bytes(got) == payload


def test_a_malformed_format_string_does_not_hide_later_problems():
    # Kills E-sandbox-254 (continue -> break after a malformed template).
    src = "def f(x, t, g):\n    a = '{'.format(x)\n    return g(g(g(t.format(1))))\n"
    problems = sandbox.static_gate(src)
    assert "malformed format string" in problems and ".format needs a string literal receiver" in problems


def _cg(tmp_path, **files):
    for name, text in files.items():
        path = tmp_path / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


@pytest.mark.parametrize("files, expected", [
    ({"cpu.max": "200000 50000\n"}, 4),                  # the period field is used, not the default
    ({"cpu.max": "100001\n"}, 2),                         # default period is exactly 100000
    ({"cpu.max": "0 100000\n"}, 1),                       # a zero quota still gets one worker
    ({"cpu__cpu.cfs_quota_us": "1\n", "cpu__cpu.cfs_period_us": "100000\n"}, 1),   # any positive quota counts
    ({"cpu__cpu.cfs_quota_us": "2\n", "cpu__cpu.cfs_period_us": "1\n"}, 2),        # any positive period counts
    ({"cpu__cpu.cfs_quota_us": "0\n", "cpu__cpu.cfs_period_us": "100000\n"}, 3),   # zero quota: fall back
    ({"cpu__cpu.cfs_quota_us": "100000\n", "cpu__cpu.cfs_period_us": "0\n"}, 3),   # zero period: fall back
])
def test_cgroup_quota_edges(tmp_path, monkeypatch, files, expected):
    # Kills E-sandbox-442/449/458 boundary and constant mutants in detect_workers.
    seen = []
    monkeypatch.setattr(sandbox.os, "sched_getaffinity", lambda pid: seen.append(pid) or {0, 1, 2}, raising=False)
    assert sandbox.detect_workers(_cg(tmp_path, **files)) == expected
    assert all(pid == 0 for pid in seen)                  # our own affinity (E-sandbox-455)


# ---------------------------------------------------------------- pool (fake workers, no wasm)

import collections  # noqa: E402
import time  # noqa: E402


class _FakeWorker:
    def __init__(self, *args):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(sandbox, "_IDLE", [])
    monkeypatch.setattr(sandbox, "_LIVE", [0])
    monkeypatch.setattr(sandbox, "_WAITING", collections.deque())
    monkeypatch.setattr(sandbox, "_INFLIGHT", collections.Counter())
    monkeypatch.setattr(sandbox, "_Worker", _FakeWorker)
    monkeypatch.setenv("QE_ARENA_WASI_WORKERS", "1")
    return monkeypatch


def test_a_full_pool_times_out_at_the_deadline_not_before_or_after(fake_pool):
    # Kills E-sandbox-524 (remaining <= 1 gives up a second early, <= -1 waits a second late).
    held = sandbox._acquire(time.monotonic() + 5)
    t = time.monotonic()
    with pytest.raises(sandbox._PoolTimeout):
        sandbox._acquire(time.monotonic() + 0.5)
    assert 0.4 <= time.monotonic() - t < 1.3
    sandbox._release(held, True)


def test_inflight_counts_never_go_negative(fake_pool):
    # Kills E-sandbox-553 (> 0 -> >= 0 / > -1): releasing a worker the session no longer counts must not
    # push its count below zero, which would let that session exceed its cap later.
    worker = sandbox._acquire(time.monotonic() + 5)
    sandbox._release(worker, True)
    sandbox._release(worker, False)            # a second release of the same worker
    assert sandbox._INFLIGHT[None] == 0 and min(sandbox._INFLIGHT.values(), default=0) >= 0
    # and the pool itself is intact: listed once, still counted live (release is idempotent)
    assert sandbox._IDLE == [worker] and sandbox._LIVE[0] == 1
    sandbox._release(worker, True)
    assert sandbox._IDLE == [worker]
    again = sandbox._acquire(time.monotonic() + 5)  # a fresh acquisition can be released again
    assert again is worker and not sandbox._IDLE
    sandbox._release(again, True)
    assert sandbox._IDLE == [worker] and sandbox._INFLIGHT[None] == 0


def test_worker_start_failure_message_is_bounded(fake_pool):
    # Kills E-sandbox-571 (truncation of the start-failure status).
    class Broken:
        def __init__(self, *args):
            raise RuntimeError("x" * 500)
    fake_pool.setattr(sandbox, "_Worker", Broken)
    status = sandbox._dispatch({})
    assert status.startswith("worker start failed: ") and len(status) == 200
    assert sandbox._LIVE[0] == 0


def test_re_debug_by_attribute_is_refused():
    # Kills S-re-debug-attr: the existing gate test passes the literal 128, never the attribute.
    assert sandbox.static_gate("import re\ndef f(p):\n    return re.compile(p, re.DEBUG)\n") == ["re.DEBUG is not allowed"]
    assert sandbox.static_gate("import re\nD = re.DEBUG\n") == ["re.DEBUG is not allowed"]


def test_a_response_for_another_request_is_refused():
    # Kills S-request-id-unchecked: a stale or forged line with someone else's id must not be accepted.
    import json
    import os
    import types
    in_r, in_w = os.pipe()
    out_r, out_w = os.pipe()
    os.set_blocking(in_w, False)
    os.write(out_w, json.dumps({"id": "not-yours", "results": []}).encode() + b"\n")
    worker = object.__new__(sandbox._Worker)
    worker.buf = bytearray()
    worker.proc = types.SimpleNamespace(stdin=os.fdopen(in_w, "wb", buffering=0),
                                        stdout=os.fdopen(out_r, "rb", buffering=0))
    assert worker.request({"id": "mine"}, time.monotonic() + 2) == "mismatched response"
    os.write(out_w, json.dumps({"id": "mine", "results": []}).encode() + b"\n")
    assert worker.request({"id": "mine"}, time.monotonic() + 2) == {"id": "mine", "results": []}


def test_rejected_results_carry_at_most_400_characters_of_reasons():
    # Kills E-sandbox-645 (truncation of the joined gate problems).
    src = "import os, sys, socket, subprocess, shutil, pathlib, ctypes, pickle, marshal, importlib\n" * 3 + \
          "def f():\n    return eval(exec(open(globals())))\n"
    [r] = sandbox.run_calls(src, "f", [[]])
    assert r.kind == "rejected" and len(r.error) == 400


# ---------------------------------------------------------------- format-path detection (39e39c1)
# The traversal-name rule now applies only to strings that reach a format path. These pin that detection.

@pytest.mark.parametrize("src", [
    "def f(t):\n    return t.format(1)\n",                          # non-literal receiver is dynamic
    "def f(t):\n    return (t + '').format(1)\n",
    "def f(x):\n    return ''.join(['{0', '}']).format(x)\n",
    "def f(x):\n    return f'{x}'.format(x)\n",                      # an f-string is not a literal receiver
])
def test_a_non_literal_receiver_is_treated_as_dynamic(src):
    assert ".format needs a string literal receiver" in sandbox.static_gate(src)


@pytest.mark.parametrize("src, problem", [
    ("def f(x, w):\n    return '{0:{1.real}}'.format(x, w)\n", "format fields may not use attribute or index lookups"),
    ("def f(x, w):\n    return '{0:{1[0]}}'.format(x, w)\n", "format fields may not use attribute or index lookups"),
    ("def f(x, w):\n    return '{0:{1}__}'.format(x, w)\n", "format string mentions '__'"),
])
def test_nested_format_specs_are_still_checked(src, problem):
    assert problem in sandbox.static_gate(src)


@pytest.mark.parametrize("src, word", [
    ("def f(x):\n    return '{0}__{0}'.format(x)\n", "__"),
    ("def f(x):\n    return '__ {}'.format(x)\n", "__"),
    ("def f(x):\n    return '{} gi_frame'.format(x)\n", "gi_frame"),
    ("def f(x):\n    return format(x, '__')\n", "__"),
])
def test_traversal_names_in_a_format_string_are_refused(src, word):
    assert f"format string mentions {word!r}" in sandbox.static_gate(src)


@pytest.mark.parametrize("src", [
    'def f(x):\n    """Mentions __doc__, f_back and gi_frame."""\n    return x\n',
    '"""Module docstring: __init__, tb_frame."""\ndef f(x):\n    return x\n',
    "def f(x):\n    return x.split('__') + ['f_locals']\n",
    "def f(x):\n    return '{0}'.format(x) + '__'\n",               # concatenated after formatting: inert
])
def test_dunder_words_outside_format_strings_are_allowed(src):
    assert sandbox.static_gate(src) == []
