"""Sandboxed execution of corpus and candidate code; the oracle stays in this process.

Layers, outermost first:

1. ``static_gate`` rejects source before it runs: imports outside the allowlist, dunder and
   introspection attributes, banned names (including ``MemoryError``, so a guest MemoryError can
   only come from the memory cap), classes, ``global``/``nonlocal``, oversize or overly deep code.
2. ``run_calls`` hands the batch to a long-lived worker process (``qe_arena/wasi_worker.py``) over a
   pipe, so the server never loads wasmtime and a worker fault cannot take down /health.
3. The worker runs CPython 3.14 compiled to WASI under wasmtime, a fresh Store and Instance per
   batch: wasmtime fuel meters every call, linear memory is capped at ``mem_mb``, the only preopen is
   a read-only stdlib zip, clocks and randomness are shadowed. Inside, the guest harness executes
   the code with restricted builtins and proxy modules that expose an audited list of names.
4. Results come back as rows the parent validates; ``compare`` runs here. Expected values never
   enter the worker or the guest.

Outcomes are deterministic: the same source and calls give the same kind, value and fuel on any
host and under any load (pinned by the corpus fingerprint test). ``fuel`` (budget exhausted) and
``resource`` (memory cap, wasm stack) are limits of the sandbox, never behaviour; ``infra`` (worker
died, wall-clock backstop) is transient and never graded. ``timeout`` is kept as an alias of
``infra`` for callers written against the subprocess sandbox.

The guest is CPython 3.14 on wasm32 while the server is 3.12 on 64-bit: pointers are 4 bytes and
``hash()`` is 32-bit. That is harmless because reference and candidate both run in the guest; never
compare a native reference with a guest result.
"""

from __future__ import annotations

import ast
import atexit
import json
import re
import string
import math
import os
import secrets
import selectors
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ALLOWED_MODULES = ("math", "re", "itertools", "functools", "collections", "string", "bisect", "heapq",
                   "statistics", "decimal", "fractions", "typing", "operator")

# Names each allowlisted module exposes to candidate code, frozen for the CPython 3.14 guest (the
# server's own 3.12 modules are irrelevant). Absent on purpose: anything that reads or writes an
# attribute or evaluates code from a runtime string (functools.update_wrapper/wraps, singledispatch,
# string.Formatter, operator.attrgetter/methodcaller, typing.get_type_hints/get_args/ForwardRef/
# evaluate_forward_ref and every TypeVar-like object that can evaluate a lazy bound) and anything
# that hands back a module (typing.sys, typing.functools, ...). tests/test_sandbox.py walks every
# public attribute reachable from these names inside the guest and asserts no module, frame, code
# object or loader is among them.
SAFE_EXPORTS: dict[str, tuple[str, ...]] = {
    "math": ("acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "comb", "copysign",
             "cos", "cosh", "degrees", "dist", "e", "erf", "erfc", "exp", "exp2", "expm1", "fabs", "factorial",
             "floor", "fma", "fmod", "frexp", "fsum", "gamma", "gcd", "hypot", "inf", "isclose", "isfinite",
             "isinf", "isnan", "isqrt", "lcm", "ldexp", "lgamma", "log", "log10", "log1p", "log2", "modf",
             "nan", "nextafter", "perm", "pi", "pow", "prod", "radians", "remainder", "sin", "sinh", "sqrt",
             "sumprod", "tan", "tanh", "tau", "trunc", "ulp"),
    "itertools": ("accumulate", "batched", "chain", "combinations", "combinations_with_replacement",
                  "compress", "count", "cycle", "dropwhile", "filterfalse", "groupby", "islice", "pairwise",
                  "permutations", "product", "repeat", "starmap", "takewhile", "tee", "zip_longest"),
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
    "typing": ("TYPE_CHECKING", "Any", "Callable", "Collection", "Container", "Dict", "Final", "FrozenSet",
               "Hashable", "Iterable", "Iterator", "List", "Literal", "Mapping", "MutableMapping",
               "MutableSequence", "MutableSet", "Optional", "Sequence", "Set", "Sized", "Tuple", "Union",
               "cast"),
    "operator": ("abs", "add", "and_", "concat", "contains", "countOf", "delitem", "eq", "floordiv", "ge",
                 "getitem", "gt", "iadd", "iand", "iconcat", "ifloordiv", "ilshift", "imatmul", "imod", "imul",
                 "index", "indexOf", "inv", "invert", "ior", "ipow", "irshift", "is_", "is_none", "is_not",
                 "is_not_none", "isub", "itemgetter", "itruediv", "ixor", "le", "length_hint", "lshift", "lt",
                 "matmul", "mod", "mul", "ne", "neg", "not_", "or_", "pos", "pow", "rshift", "setitem", "sub",
                 "truediv", "truth", "xor"),
}

SAFE_BUILTINS = (
    "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate", "filter", "float", "format",
    "frozenset", "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "callable",
    "ArithmeticError", "AssertionError", "AttributeError", "Exception", "IndexError", "KeyError",
    "LookupError", "NotImplementedError", "OverflowError", "RecursionError",
    "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
    "True", "False", "None", "NotImplemented",
)

BANNED_NAMES = frozenset({
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "breakpoint", "input", "help", "memoryview", "exit", "quit", "MemoryError",
})
# Introspection attributes that reach frames, code objects or tracebacks without a dunder, and the
# string-driven attribute or evaluation helpers that are kept out of SAFE_EXPORTS (defence in depth).
DENIED_ATTR_PREFIXES = ("_", "gi_", "cr_", "ag_", "f_", "tb_", "co_")
DENIED_ATTRS = frozenset({"mro", "format_map", "get_field", "vformat", "convert_field", "format_field",
                          "get_value", "with_traceback", "attrgetter", "methodcaller", "get_type_hints",
                          "get_args", "get_origin", "evaluate", "evaluate_forward_ref", "evaluate_bound",
                          "evaluate_constraints", "evaluate_default"})
# str.format resolves "{0.attr}" and "{0[key]}" at runtime, outside the AST checks, and reaches frames
# and module globals through dunders ("{0.__traceback__.tb_frame.f_back.f_locals}"). So a string
# literal may not mention a traversal name, .format needs a literal receiver whose fields are plain
# positions or names, and format() needs a literal spec.
TRAVERSAL_TOKENS = ("__", "tb_frame", "tb_next", "f_back", "f_locals", "f_globals", "f_builtins", "f_code",
                    "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code")
# "__" anywhere; the frame attribute names as whole words, so "lines_of_code" in a docstring is fine
_TRAVERSAL_RE = re.compile(r"__|(?<![A-Za-z0-9])(?:" + "|".join(TRAVERSAL_TOKENS[1:]) + r")(?![A-Za-z0-9_])")
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

MAX_SOURCE_BYTES = 6000
MAX_AST_DEPTH = 100
MAX_RESULT_BYTES = 960 * 1024
MAX_VALUE_BYTES = 128 * 1024
MAX_RESPONSE_BYTES = 4 << 20

DEFAULT_MEM_MB = 256              # linear-memory cap per instance
FUEL_PER_CALL = 100_000_000       # ~90x the most expensive corpus call (1.1 M)
MODULE_FUEL = 200_000_000         # executing the candidate's module body
FUEL_PER_SECOND = 200_000_000     # only to translate a legacy per_call_timeout_s into fuel
WORKER_STARTUP_S = 180.0          # compiling python.wasm takes ~5 s at --cpus=2; generous for loaded hosts
BATCH_BACKSTOP_S = 300.0          # parent-side wait for one batch before the worker is replaced

KINDS = ("ok", "exc", "fuel", "resource", "crash", "skipped", "rejected", "infra", "timeout")
WORKER_KINDS = frozenset({"ok", "exc", "fuel", "resource", "crash", "skipped", "infra"})


@dataclass
class CallResult:
    """One call's outcome. ``fuel`` is what the call consumed (0 when it never ran)."""
    kind: Literal["ok", "exc", "fuel", "resource", "crash", "skipped", "rejected", "infra", "timeout"]
    value: Any = None
    error: str = ""
    fuel: int = 0


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


def _format_fields(template: str) -> list[str]:
    """Field names in a format template, nested specs included; raises ValueError if malformed."""
    names = []
    for _, field, spec, _ in string.Formatter().parse(template):
        if field is not None:
            names.append(field)
        if spec:
            names.extend(_format_fields(spec))
    return names


def _check_formatting(tree: ast.AST) -> list[str]:
    problems = []
    allowed_format_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found = _TRAVERSAL_RE.search(node.value)
            if found:
                problems.append(f"string literal mentions {found.group(0)!r}")
        elif isinstance(node, ast.Attribute) and node.attr == "format":
            receiver = node.value
            if not (isinstance(receiver, ast.Constant) and isinstance(receiver.value, str)):
                problems.append(".format needs a string literal receiver")
                continue
            try:
                fields = _format_fields(receiver.value)
            except ValueError:
                problems.append("malformed format string")
                continue
            if any("." in f or "[" in f for f in fields):
                problems.append("format fields may not use attribute or index lookups")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "format":
            spec_ok = len(node.args) == 1 or (
                len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str))
            if spec_ok and not node.keywords:
                allowed_format_names.add(node.func)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "format" and node not in allowed_format_names:
            problems.append("format() needs a string literal spec and may not be passed around")
    return problems


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
    for problem in [p for node in ast.walk(tree) for p in _check_node(node)] + _check_formatting(tree):
        if problem not in problems:
            problems.append(problem)
    return problems


# --------------------------------------------------------------------------- worker pool

PACKAGE_PARENT = Path(__file__).resolve().parents[1]


def runtime_root() -> Path:
    """Where python.wasm and lib/python314.zip live (built by scripts/build_wasi_runtime.py)."""
    configured = os.environ.get("QE_ARENA_WASI_ROOT")
    if configured:
        return Path(configured)
    local = PACKAGE_PARENT / "runtime"
    return local if (local / "python.wasm").exists() else Path("/opt/qe-arena/wasi")


class _Worker:
    def __init__(self) -> None:
        env = dict(os.environ, QE_ARENA_WASI_ROOT=str(runtime_root()), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join(filter(None, [str(PACKAGE_PARENT), os.environ.get("PYTHONPATH")])))
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-B", "-m", "qe_arena.wasi_worker"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=None, env=env, close_fds=True)
        self.buf = bytearray()
        ready = self.read(time.monotonic() + WORKER_STARTUP_S)
        if not (isinstance(ready, dict) and ready.get("ready")):
            self.close()
            raise RuntimeError(f"sandbox worker did not start: {ready!r}")

    def read(self, deadline: float) -> Any:
        """One JSON line from the worker, or a status string on timeout, EOF or overflow."""
        fd = self.proc.stdout.fileno()
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            while True:
                nl = self.buf.find(b"\n")
                if nl >= 0:
                    line = bytes(self.buf[:nl])
                    del self.buf[:nl + 1]
                    try:
                        return json.loads(line.decode("utf-8", "surrogatepass"))
                    except ValueError:
                        return "bad response"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "timeout"
                if not sel.select(remaining):
                    continue
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    return "exited"
                self.buf += chunk
                if len(self.buf) > MAX_RESPONSE_BYTES:
                    return "overflow"

    def request(self, payload: dict, deadline: float) -> Any:
        try:
            self.proc.stdin.write(json.dumps(payload, allow_nan=True).encode("utf-8", "surrogatepass") + b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return "exited"
        return self.read(deadline)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL does not fail on POSIX
            pass
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except OSError:
                pass


MAX_WORKERS = 4
CGROUP_ROOT = Path("/sys/fs/cgroup")


def detect_workers(cgroup_root: Path = CGROUP_ROOT) -> int:
    """One worker per CPU the container may actually use, between 1 and MAX_WORKERS.

    ``docker run --cpus=2`` sets a CFS quota but leaves every host CPU visible, so os.cpu_count()
    and the affinity mask say 16 on ruos. Read the quota first: cgroup v2 ``cpu.max`` ("max 100000"
    or "200000 100000"), then cgroup v1 ``cpu.cfs_quota_us``/``cpu.cfs_period_us``, then the
    affinity mask, then os.cpu_count(). A fractional quota rounds up.
    """
    cpus = None
    try:
        fields = (cgroup_root / "cpu.max").read_text().split()
        if fields and fields[0] != "max":
            cpus = int(fields[0]) / int(fields[1] if len(fields) > 1 else 100_000)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    if cpus is None:
        try:
            quota = int((cgroup_root / "cpu" / "cpu.cfs_quota_us").read_text())
            period = int((cgroup_root / "cpu" / "cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                cpus = quota / period
        except (OSError, ValueError):
            pass
    if cpus is None:
        try:
            cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpus = os.cpu_count() or 1
    return min(MAX_WORKERS, max(1, math.ceil(cpus)))


_DETECTED: list[int] = []


def _pool_size() -> int:
    """QE_ARENA_WASI_WORKERS overrides (at least 1); otherwise ``detect_workers()``, read once."""
    override = os.environ.get("QE_ARENA_WASI_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if not _DETECTED:
        _DETECTED.append(detect_workers())
    return _DETECTED[0]


_POOL = threading.Condition()
_IDLE: list[_Worker] = []
_LIVE = [0]


def _acquire() -> _Worker:
    with _POOL:
        while not _IDLE and _LIVE[0] >= _pool_size():
            _POOL.wait()
        if _IDLE:
            return _IDLE.pop()
        _LIVE[0] += 1
    try:
        return _Worker()
    except BaseException:
        with _POOL:
            _LIVE[0] -= 1
            _POOL.notify()
        raise


def _release(worker: _Worker, healthy: bool) -> None:
    if not healthy:
        worker.close()
    with _POOL:
        if healthy:
            _IDLE.append(worker)
        else:
            _LIVE[0] -= 1
        _POOL.notify()


def _dispatch(payload: dict) -> Any:
    """Send one request; return the response dict or a status string. Replaces a worker that fails."""
    worker = _acquire()
    response: Any = "exited"
    try:
        response = worker.request(dict(payload, id=secrets.token_hex(8)), time.monotonic() + BATCH_BACKSTOP_S)
        return response
    finally:
        _release(worker, isinstance(response, dict) and "error" not in response)


def warm() -> None:
    """Start every worker now (each compiles python.wasm) so the first episode does not wait."""
    workers = [_acquire() for _ in range(_pool_size())]
    for worker in workers:
        _release(worker, True)


def shutdown() -> None:
    with _POOL:
        workers, _IDLE[:] = list(_IDLE), []
        _LIVE[0] -= len(workers)
        _POOL.notify_all()
    for worker in workers:
        worker.close()


atexit.register(shutdown)


def _parse_results(response: Any, n: int) -> list[CallResult]:
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise ValueError(f"no results: {response!r}"[:200])
    rows = response["results"]
    if len(rows) != n:
        raise ValueError("result count mismatch")
    out = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"k", "v", "e", "f"}:
            raise ValueError("bad result row")
        kind, error, fuel = row.get("k"), row.get("e", ""), row.get("f", 0)
        if kind not in WORKER_KINDS or not isinstance(error, str) or type(fuel) is not int or fuel < 0:
            raise ValueError("bad result kind")
        if kind == "exc" and not isinstance(row.get("v"), str):
            raise ValueError("bad exception name")
        value = row.get("v") if kind in ("ok", "exc") else None
        out.append(CallResult(kind=kind, value=value, error=error[:200], fuel=fuel))
    return out


def run_calls(source: str, entry: str, calls: list[list], *, timeout_s: float | None = None,
              per_call_timeout_s: float | None = None, mem_mb: int = DEFAULT_MEM_MB, fuel: int | None = None,
              _skip_gate: bool = False) -> list[CallResult]:
    """Call ``entry(*args)`` for each args list in ``calls`` in one sandboxed guest instance.

    Each call may consume ``fuel`` units (default ``FUEL_PER_CALL``) and the module body
    ``MODULE_FUEL``. ``per_call_timeout_s`` is accepted for compatibility and converted at
    ``FUEL_PER_SECOND``; ``timeout_s`` is accepted and ignored (fuel bounds every call and a
    wall-clock backstop bounds the instance). ``_skip_gate`` exists only so tests can prove the
    runtime layers on their own.
    """
    del timeout_s
    if not all(isinstance(c, (list, tuple)) for c in calls):
        raise TypeError("each call must be a list of positional arguments")
    calls = [list(c) for c in calls]
    if not calls:
        return []
    if not _skip_gate:
        problems = static_gate(source)
        if problems:
            return [CallResult("rejected", None, "; ".join(problems)[:400]) for _ in calls]
    if fuel is None:
        fuel = FUEL_PER_CALL if per_call_timeout_s is None else max(1, int(per_call_timeout_s * FUEL_PER_SECOND))
    payload = {
        "source": source, "entry": entry, "calls": calls, "nonce": secrets.token_hex(16),
        "fuel": int(fuel), "module_fuel": MODULE_FUEL, "mem_mb": int(mem_mb),
        "modules": SAFE_EXPORTS, "builtins": SAFE_BUILTINS,
        "max_value_bytes": MAX_VALUE_BYTES, "max_total_bytes": MAX_RESULT_BYTES,
    }
    status = ""
    for _ in range(2):  # a dead worker is replaced and the batch retried once
        response = _dispatch(payload)
        try:
            return _parse_results(response, len(calls))
        except ValueError as exc:
            status = str(exc)
    return [CallResult("infra", None, f"sandbox worker: {status}"[:200]) for _ in calls]


def _probe_guest(kind: str = "surface", mem_mb: int = DEFAULT_MEM_MB) -> dict:
    """Ask the guest harness to describe its own surface (test hook; runs no candidate code)."""
    response = _dispatch({"probe": kind, "nonce": secrets.token_hex(16), "modules": SAFE_EXPORTS,
                          "mem_mb": mem_mb})
    return response.get("probe", {}) if isinstance(response, dict) else {"error": response}


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
    including fuel, resource, crash, infra and rejected on either side, is a mismatch. Fuel is never
    compared: it depends on what ran earlier in the same instance."""
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
