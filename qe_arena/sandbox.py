"""Out-of-process execution of corpus and candidate code; the oracle stays in the parent.

Layers, outermost first:

1. ``static_gate`` rejects source before it runs: imports outside the allowlist, dunder and
   introspection attributes, banned builtins, classes, ``global``/``nonlocal``, oversize or
   overly deep code.
2. ``run_calls`` runs one fresh ``python -S -s -P`` child per batch with an environment holding only
   ``PYTHONHASHSEED=0``, a temporary working directory and its own session (the whole process group
   is killed on timeout). That is ``-I`` minus ``-E``: ``-E`` would discard the hash seed, and with
   randomized string hashing ``list(set(words))`` orders differently in the reference process and the
   candidate process, so identical code could score differently. The parent controls the whole
   environment, so ignoring it buys nothing.
3. The child sets rlimits on itself (address space, CPU, file size 0, open files, core 0) *before*
   any candidate code is compiled, sends fds 1 and 2 to /dev/null, and executes the code with a
   restricted ``__builtins__`` whose ``__import__`` only returns pre-built proxy modules exposing an
   audited list of names. Each call has its own interval timer.
4. The child writes one result line, prefixed by a per-run random nonce, to a private dup of the
   original stdout. The parent reads at most 1 MiB, accepts exactly one nonce line, validates its
   shape, and compares values itself. Expected values never enter the child.

Rlimits that the platform cannot apply (macOS refuses RLIMIT_AS) are skipped; on Linux they hold.
"""

from __future__ import annotations

import ast
import json
import math
import os
import secrets
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

ALLOWED_MODULES = ("math", "re", "itertools", "functools", "collections", "string", "bisect", "heapq",
                   "statistics", "decimal", "fractions")


def _public(module_name: str) -> tuple[str, ...]:
    module = __import__(module_name)
    return tuple(sorted(n for n in dir(module) if not n.startswith("_")))


# Names each allowlisted module exposes to candidate code. Anything that can read or write an
# attribute by a runtime string (functools.update_wrapper/wraps, singledispatch's annotation eval,
# string.Formatter.get_field) or that hands back another module is deliberately absent.
SAFE_EXPORTS: dict[str, tuple[str, ...]] = {
    "math": _public("math"),
    "itertools": _public("itertools"),
    "re": ("A", "ASCII", "DOTALL", "I", "IGNORECASE", "M", "MULTILINE", "NOFLAG", "S", "VERBOSE", "X",
           "Match", "Pattern", "compile", "error", "escape", "findall", "finditer", "fullmatch", "match",
           "search", "split", "sub", "subn"),
    "functools": ("cache", "cmp_to_key", "lru_cache", "partial", "reduce"),
    "collections": ("ChainMap", "Counter", "OrderedDict", "defaultdict", "deque", "namedtuple"),
    "string": ("Template", "ascii_letters", "ascii_lowercase", "ascii_uppercase", "capwords", "digits",
               "hexdigits", "octdigits", "printable", "punctuation", "whitespace"),
    "bisect": ("bisect", "bisect_left", "bisect_right", "insort", "insort_left", "insort_right"),
    "heapq": ("heapify", "heappop", "heappush", "heappushpop", "heapreplace", "merge", "nlargest",
              "nsmallest"),
    "statistics": ("StatisticsError", "correlation", "covariance", "fmean", "geometric_mean",
                   "harmonic_mean", "linear_regression", "mean", "median", "median_grouped", "median_high",
                   "median_low", "mode", "multimode", "pstdev", "pvariance", "quantiles", "stdev",
                   "variance"),
    "decimal": ("BasicContext", "Clamped", "Context", "Decimal", "DecimalException", "DefaultContext",
                "DivisionByZero", "ExtendedContext", "FloatOperation", "Inexact", "InvalidOperation",
                "Overflow", "ROUND_05UP", "ROUND_CEILING", "ROUND_DOWN", "ROUND_FLOOR", "ROUND_HALF_DOWN",
                "ROUND_HALF_EVEN", "ROUND_HALF_UP", "ROUND_UP", "Rounded", "Subnormal", "Underflow",
                "getcontext", "localcontext", "setcontext"),
    "fractions": ("Fraction",),
}

SAFE_BUILTINS = (
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "callable",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError", "KeyError",
    "LookupError", "MemoryError", "NotImplementedError", "OverflowError", "RecursionError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
    "True", "False", "None", "NotImplemented",
)

BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "breakpoint", "input", "help", "memoryview", "exit", "quit",
})
# Introspection attributes that reach frames, code objects or tracebacks without a dunder.
DENIED_ATTR_PREFIXES = ("_", "gi_", "cr_", "ag_", "f_", "tb_", "co_")
DENIED_ATTRS = frozenset({"mro", "format_map", "get_field", "vformat", "convert_field", "format_field",
                          "get_value", "with_traceback"})
REJECTED_NODES = {
    ast.ClassDef: "class definitions are not allowed",
    ast.Global: "global statements are not allowed",
    ast.Nonlocal: "nonlocal statements are not allowed",
    ast.AsyncFunctionDef: "async functions are not allowed",
    ast.Await: "await is not allowed",
    ast.AsyncFor: "async for is not allowed",
    ast.AsyncWith: "async with is not allowed",
    ast.Match: "match statements are not allowed",
}
if hasattr(ast, "TypeAlias"):
    REJECTED_NODES[ast.TypeAlias] = "type aliases are not allowed"

CHILD_ENV = {"PYTHONHASHSEED": "0"}
MAX_SOURCE_BYTES = 6000
MAX_AST_DEPTH = 100
MAX_STDOUT_BYTES = 1 << 20
MAX_RESULT_BYTES = 960 * 1024
MAX_VALUE_BYTES = 128 * 1024
KILL_GRACE_S = 0.5


@dataclass
class CallResult:
    kind: Literal["ok", "exc", "timeout", "crash", "rejected"]
    value: Any = None
    error: str = ""


# --------------------------------------------------------------------------- static gate

def _is_dunder(name: str | None) -> bool:
    return bool(name) and name.startswith("__")


def _ast_depth(tree: ast.AST) -> int:
    deepest, stack = 0, [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        deepest = max(deepest, depth)
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return deepest


def _check_node(node: ast.AST) -> list[str]:
    problems = []
    for node_type, message in REJECTED_NODES.items():
        if isinstance(node, node_type):
            problems.append(message)
    if isinstance(node, ast.Name):
        if _is_dunder(node.id):
            problems.append(f"dunder name {node.id!r}")
        elif node.id in BANNED_NAMES:
            problems.append(f"banned name {node.id!r}")
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith(DENIED_ATTR_PREFIXES) or node.attr in DENIED_ATTRS:
            problems.append(f"attribute {node.attr!r} is not allowed")
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                problems.append(f"import of {alias.name!r} is not allowed")
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level or module not in ALLOWED_MODULES:
            problems.append(f"import from {'.' * node.level + module!r} is not allowed")
        else:
            for alias in node.names:
                if alias.name not in SAFE_EXPORTS[module]:
                    problems.append(f"{module}.{alias.name} is not available")
    elif isinstance(node, ast.FunctionDef) and getattr(node, "type_params", None):
        problems.append("type parameters are not allowed")
    for field in ("name", "arg", "asname"):
        value = getattr(node, field, None)
        if isinstance(value, str) and _is_dunder(value):
            problems.append(f"dunder identifier {value!r}")
    if isinstance(node, ast.alias) and _is_dunder(node.name):
        problems.append(f"dunder identifier {node.name!r}")
    return problems


def static_gate(source: str) -> list[str]:
    """Return the reasons ``source`` may not run; an empty list means it passed."""
    if not isinstance(source, str):
        return ["source must be a string"]
    size = len(source.encode("utf-8", "surrogatepass"))
    if size > MAX_SOURCE_BYTES:
        return [f"source is {size} bytes; the limit is {MAX_SOURCE_BYTES}"]
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return [f"does not parse: {type(exc).__name__}: {exc}"[:200]]
    if _ast_depth(tree) > MAX_AST_DEPTH:
        return [f"syntax tree deeper than {MAX_AST_DEPTH}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        for problem in _check_node(node):
            if problem not in problems:
                problems.append(problem)
    return problems


# --------------------------------------------------------------------------- child harness

CHILD_SRC = r'''
import builtins, copy, json, os, signal, sys, time, types
def _main():
    raw = sys.stdin.buffer.read()
    req = json.loads(raw)
    nonce = req["nonce"]
    out_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    started = time.monotonic()
    deadline = started + req["budget_s"]
    per_call = req["per_call_s"]

    def emit(obj):
        data = (nonce + " " + json.dumps(obj, allow_nan=True) + "\n").encode("utf-8", "surrogatepass")
        view = memoryview(data)
        while view:
            view = view[os.write(out_fd, view):]

    real = {}
    for name in req["modules"]:
        real[name] = __import__(name)
    applied = {}
    try:
        import resource
    except ImportError:
        resource = None
    if resource is not None:
        for name, value in req["limits"].items():
            res = getattr(resource, name, None)
            if res is None:
                applied[name] = "unsupported"
                continue
            try:
                resource.setrlimit(res, (value[0], value[1]))
                applied[name] = list(resource.getrlimit(res))
            except (ValueError, OSError) as exc:
                applied[name] = "failed: " + type(exc).__name__
    if req.get("probe"):
        emit({"applied": applied, "env": sorted(os.environ), "cwd_entries": len(os.listdir(".")),
              "flags": {"no_site": sys.flags.no_site, "no_user_site": sys.flags.no_user_site,
                        "safe_path": sys.flags.safe_path, "hash_randomization": sys.flags.hash_randomization},
              "path": sys.path})
        return

    proxies = {}
    for name, exports in req["modules"].items():
        proxy = types.ModuleType("_arena_safe_" + name)
        for attr in exports:
            if hasattr(real[name], attr):
                setattr(proxy, attr, getattr(real[name], attr))
        proxies[name] = proxy
    helper_ns = {"__builtins__": {"ImportError": ImportError, "repr": repr, "type": type, "str": str}, "_M": proxies}
    exec("def __import__(name, globals=None, locals=None, fromlist=(), level=0):\n"
         "    if level == 0 and type(name) is str and name in _M:\n"
         "        return _M[name]\n"
         "    raise ImportError('import of ' + repr(name) + ' is not allowed')\n"
         "def print(*args, **kwargs):\n"
         "    return None\n", helper_ns)
    safe = {name: getattr(builtins, name) for name in req["builtins"]}
    safe["__import__"] = helper_ns["__import__"]
    safe["print"] = helper_ns["print"]

    class _Timeout(BaseException):
        pass
    armed = [False]
    fired = [False]
    def on_alarm(signum, frame):
        if armed[0]:
            armed[0] = False
            fired[0] = True
            raise _Timeout()
    signal.signal(signal.SIGALRM, on_alarm)

    def guarded(thunk, limit):
        fired[0] = False
        outcome = None
        try:
            armed[0] = True
            signal.setitimer(signal.ITIMER_REAL, max(limit, 0.001))
            try:
                outcome = ("ok", thunk())
            except _Timeout:
                outcome = ("timeout", None)
            except BaseException as exc:
                outcome = ("exc", exc)
            armed[0] = False
        except _Timeout:
            armed[0] = False
            outcome = ("timeout", None)
        signal.setitimer(signal.ITIMER_REAL, 0)
        if fired[0]:
            return ("timeout", None)
        return outcome

    def describe(exc):
        try:
            return (type(exc).__name__ + ": " + str(exc))[:200]
        except BaseException:
            return type(exc).__name__

    def normalize(value, depth, budget):
        budget[0] += 1
        if depth > 100 or budget[0] > 200000:
            raise ValueError("too large")
        t = type(value)
        if value is None or t is bool or t is int or t is float or t is str:
            return value
        if t is list or t is tuple:
            return [normalize(v, depth + 1, budget) for v in value]
        if t is dict:
            out = {}
            for k, v in value.items():
                if type(k) is not str:
                    raise ValueError("non-str key")
                out[k] = normalize(v, depth + 1, budget)
            return out
        raise ValueError("type " + t.__name__)

    n = len(req["calls"])
    def fill(kind, error=""):
        return [{"k": kind, "e": error} for _ in range(n)]

    ns = {"__builtins__": safe, "__name__": "candidate"}
    try:
        code = compile(req["source"], "<candidate>", "exec")
    except BaseException as exc:
        emit(fill("crash", "compile: " + describe(exc)))
        return
    kind, value = guarded(lambda: exec(code, ns), min(per_call * 4, deadline - time.monotonic()))
    if kind != "ok":
        if kind == "timeout":
            emit(fill("timeout", "module timeout"))
        else:
            emit(fill("crash", "module: " + describe(value)))
        return
    fn = ns.get(req["entry"])
    if not callable(fn):
        emit(fill("crash", "entry " + repr(req["entry"]) + " is not a function"))
        return

    results, total = [], 0
    for call in req["calls"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.append({"k": "timeout", "e": "batch budget exhausted"})
            continue
        args = copy.deepcopy(call)
        kind, value = guarded(lambda: fn(*args), min(per_call, remaining))
        if kind == "ok":
            try:
                value = normalize(value, 0, [0])
                size = len(json.dumps(value, allow_nan=True))
            except (ValueError, RecursionError, MemoryError, TypeError):
                results.append({"k": "crash", "e": "unserializable"})
                continue
            if size > req["max_value_bytes"] or total + size > req["max_total_bytes"]:
                results.append({"k": "crash", "e": "output too large"})
                continue
            total += size
            results.append({"k": "ok", "v": value})
        elif kind == "exc":
            results.append({"k": "exc", "v": type(value).__name__, "e": describe(value)})
        else:
            results.append({"k": "timeout", "e": "per-call timeout"})
    emit(results)

try:
    _main()
except BaseException:
    pass
os._exit(0)
'''


def _child_limits(mem_mb: int, timeout_s: float) -> dict[str, list[int]]:
    cpu = math.ceil(timeout_s) + 1
    mem = int(mem_mb) << 20
    return {
        "RLIMIT_AS": [mem, mem],
        "RLIMIT_CPU": [cpu, cpu + 1],
        "RLIMIT_FSIZE": [0, 0],
        "RLIMIT_NOFILE": [16, 16],
        "RLIMIT_CORE": [0, 0],
    }


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL does not fail on POSIX
        pass


def _exchange(payload: bytes, deadline: float) -> tuple[bytes, str]:
    """Run the child; return (stdout bytes, status) with status in ok/timeout/overflow/exit:<rc>."""
    with tempfile.TemporaryDirectory(prefix="qe-arena-") as cwd:
        proc = subprocess.Popen(
            [sys.executable, "-S", "-s", "-P", "-c", CHILD_SRC],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=CHILD_ENV, cwd=cwd, start_new_session=True, close_fds=True)

        def feed() -> None:
            try:
                proc.stdin.write(payload)
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        writer = threading.Thread(target=feed, daemon=True)
        writer.start()
        buf, status = bytearray(), "ok"
        fd = proc.stdout.fileno()
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(fd, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timeout"
                        break
                    if not sel.select(remaining):
                        continue
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    if len(buf) + len(chunk) > MAX_STDOUT_BYTES:
                        status = "overflow"
                        break
                    buf += chunk
            if status == "ok":
                try:
                    proc.wait(timeout=max(0.05, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    status = "timeout"
        finally:
            _kill_group(proc)
            writer.join(timeout=1)
            for pipe in (proc.stdin, proc.stdout):
                try:
                    pipe.close()
                except OSError:
                    pass
        if status == "ok" and proc.returncode != 0:
            status = f"exit:{proc.returncode}"
        return bytes(buf), status


def _result_line(buf: bytes, nonce: str) -> Any:
    prefix = (nonce + " ").encode()
    lines = [line for line in buf.split(b"\n") if line.startswith(prefix)]
    if len(lines) != 1:
        raise ValueError(f"expected one result line, found {len(lines)}")
    return json.loads(lines[0][len(prefix):].decode("utf-8", "surrogatepass"))


def _parse_results(data: Any, n: int) -> list[CallResult]:
    if not isinstance(data, list) or len(data) != n:
        raise ValueError("result shape mismatch")
    out = []
    for row in data:
        if not isinstance(row, dict) or set(row) - {"k", "v", "e"}:
            raise ValueError("bad result row")
        kind, error = row.get("k"), row.get("e", "")
        if kind not in ("ok", "exc", "timeout", "crash") or not isinstance(error, str):
            raise ValueError("bad result kind")
        if kind == "exc" and not isinstance(row.get("v"), str):
            raise ValueError("bad exception name")
        value = row.get("v") if kind in ("ok", "exc") else None
        out.append(CallResult(kind=kind, value=value, error=error[:200]))
    return out


def _launch(request: dict, timeout_s: float) -> tuple[Any, str]:
    nonce = secrets.token_hex(16)
    request = dict(request, nonce=nonce)
    payload = json.dumps(request, allow_nan=True).encode("utf-8", "surrogatepass")
    buf, status = _exchange(payload, time.monotonic() + timeout_s + KILL_GRACE_S)
    try:
        return _result_line(buf, nonce), status
    except (ValueError, UnicodeDecodeError) as exc:
        return None, status if status != "ok" else f"bad output: {exc}"


def run_calls(source: str, entry: str, calls: list[list], *, timeout_s: float = 3.0,
              per_call_timeout_s: float = 0.5, mem_mb: int = 256,
              _skip_gate: bool = False) -> list[CallResult]:
    """Call ``entry(*args)`` for each args list in ``calls`` inside one sandboxed child process.

    ``timeout_s`` bounds the whole batch (the child stops starting calls at the budget and the
    parent kills the process group shortly after); ``per_call_timeout_s`` bounds each call.
    ``_skip_gate`` exists only so tests can prove the runtime layer on its own.
    """
    if not all(isinstance(c, (list, tuple)) for c in calls):
        raise TypeError("each call must be a list of positional arguments")
    calls = [list(c) for c in calls]
    if not calls:
        return []
    if not _skip_gate:
        problems = static_gate(source)
        if problems:
            return [CallResult("rejected", None, "; ".join(problems)[:400]) for _ in calls]
    request = {
        "source": source, "entry": entry, "calls": calls,
        "budget_s": max(0.05, timeout_s - 0.15), "per_call_s": per_call_timeout_s,
        "modules": SAFE_EXPORTS, "builtins": SAFE_BUILTINS,
        "limits": _child_limits(mem_mb, timeout_s),
        "max_value_bytes": MAX_VALUE_BYTES, "max_total_bytes": MAX_RESULT_BYTES,
    }
    data, status = _launch(request, timeout_s)
    if data is not None and status == "ok":
        try:
            return _parse_results(data, len(calls))
        except ValueError as exc:
            status = f"bad output: {exc}"
    if status == "timeout" or status == f"exit:{-signal.SIGXCPU}":
        return [CallResult("timeout", None, "batch timeout") for _ in calls]
    return [CallResult("crash", None, f"sandbox: {status}") for _ in calls]


def _probe_limits(mem_mb: int = 256, timeout_s: float = 3.0) -> dict:
    """Report the child's rlimits, environment and flags (test hook; runs no candidate code)."""
    request = {"source": "", "entry": "", "calls": [], "budget_s": timeout_s, "per_call_s": 0.5,
               "modules": SAFE_EXPORTS, "builtins": SAFE_BUILTINS, "probe": True,
               "limits": _child_limits(mem_mb, timeout_s)}
    data, _ = _launch(request, timeout_s)
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- comparison

def _equal(a: Any, b: Any) -> bool:
    ta, tb = type(a), type(b)
    if ta is not tb:
        return False
    if ta is float:
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if ta is list:
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if ta is dict:
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if ta in (int, str, bool, type(None)):
        return a == b
    return False


def compare(expected: CallResult, actual: CallResult) -> bool:
    """True when two results agree: ok/ok by strict JSON value equality (bool is not int, int is not
    float, floats within rel 1e-9 / abs 1e-12), exc/exc by exception type name. Anything else,
    including timeout, crash and rejected on either side, is a mismatch."""
    if expected.kind == "ok" and actual.kind == "ok":
        try:
            return _equal(_jsonish(expected.value), _jsonish(actual.value))
        except (RecursionError, ValueError):
            return False
    if expected.kind == "exc" and actual.kind == "exc":
        return isinstance(expected.value, str) and expected.value == actual.value
    return False


def _jsonish(value: Any) -> Any:
    """Normalize a caller-built value (tuples become lists) the way the child normalizes results."""
    if isinstance(value, tuple):
        return [_jsonish(v) for v in value]
    if type(value) is list:
        return [_jsonish(v) for v in value]
    if type(value) is dict:
        return {k: _jsonish(v) for k, v in value.items()}
    return value
