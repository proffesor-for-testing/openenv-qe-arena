# Runs INSIDE python.wasm (CPython 3.14, WASI preview1). The host passes this file with -c and the
# request as argv[1]. It is never imported by the server.
#
# No rlimits, signals or timers: the host meters every call with wasmtime fuel and caps linear
# memory. Protocol on fd 1, one os.write per line so the host sees call boundaries exactly; every
# line carries the per-run nonce the host generated, and the host ignores or rejects anything else:
#   \x01B <nonce>\n          a call starts      -> host sets fuel to this call's budget
#   \x01E <nonce>\n          that call ended    -> host records fuel used, refuels the harness
#   \x01R <nonce> <json>\n   that call's result row
#
# Frame hygiene: candidate code runs below exactly two frames, the closure returned by _setup()
# and the root module frame, whose globals are cleared before the closure runs. The closure's
# variables are the candidate's own namespace plus opaque objects: the nonce exists only inside
# itertools.cycle objects (no attribute or repr exposes their contents) and the pending inputs only
# inside a list iterator. The request, sys.argv and sys.orig_argv are gone by then.
import builtins, copy, itertools, json, os, sys, types


def _setup():
    req = json.loads(sys.argv[1])
    sys.argv[:] = ["python"]
    sys.orig_argv[:] = []
    nonce = req.pop("nonce").encode("ascii")
    begin = itertools.cycle((b"\x01B " + nonce + b"\n",))
    end = itertools.cycle((b"\x01E " + nonce + b"\n",))
    head = itertools.cycle((b"\x01R " + nonce + b" ",))
    del nonce
    write, dumps, deepcopy = os.write, json.dumps, copy.deepcopy

    def emit(marker, payload=b""):
        view = memoryview(next(marker) + payload)
        while view:
            view = view[write(1, view):]

    def row(obj):
        emit(head, dumps(obj, allow_nan=True).encode("utf-8", "surrogatepass") + b"\n")

    real = {name: __import__(name) for name in req["modules"]}
    if req.get("probe"):
        row(_probe(req["probe"], real, req["modules"]))
        return None
    proxies = {}
    for name, exports in req["modules"].items():
        proxy = types.ModuleType("_arena_safe_" + name)
        for attr in exports:
            if hasattr(real[name], attr):
                setattr(proxy, attr, getattr(real[name], attr))
        proxies[name] = proxy
    helper_ns = {"__builtins__": {"ImportError": ImportError, "repr": repr, "type": type, "str": str},
                 "_M": proxies}
    exec("def __import__(name, globals=None, locals=None, fromlist=(), level=0):\n"
         "    if level == 0 and type(name) is str and name in _M:\n"
         "        return _M[name]\n"
         "    raise ImportError('import of ' + repr(name) + ' is not allowed')\n"
         "def print(*args, **kwargs):\n"
         "    return None\n", helper_ns)
    safe = {name: getattr(builtins, name) for name in req["builtins"]}
    safe["__import__"] = helper_ns["__import__"]
    safe["print"] = helper_ns["print"]
    del real, proxies, helper_ns

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

    def classify(exc):
        # Allocation failure inside the capped linear memory surfaces as MemoryError. The static gate
        # bans the MemoryError name, so candidate code cannot raise one on purpose.
        if isinstance(exc, MemoryError):
            return {"k": "resource", "e": "memory: " + describe(exc)}
        return {"k": "exc", "v": type(exc).__name__, "e": describe(exc)}

    source, entry = req["source"], req["entry"]
    calls = iter(req["calls"][req["start"]:])
    max_value, max_total = req["max_value_bytes"], req["max_total_bytes"]
    del req

    def run():
        ns = {"__builtins__": safe, "__name__": "candidate"}
        emit(begin)  # the module body is metered with the module budget
        try:
            exec(compile(source, "<candidate>", "exec"), ns)
            module_err = None
        except BaseException as exc:
            module_err = exc
        emit(end)
        if module_err is not None:
            kind = "resource" if isinstance(module_err, MemoryError) else "crash"
            row({"k": kind, "e": "module: " + describe(module_err)})
            return
        fn = ns.get(entry)
        if not callable(fn):
            row({"k": "crash", "e": "entry " + repr(entry) + " is not a function"})
            return
        row({"k": "ok"})
        total = 0
        for call in calls:
            args = deepcopy(call)
            del call
            write(1, next(begin))  # one write: the host's fd_write always takes the whole marker
            try:
                value = fn(*args)
                err = None
            except BaseException as exc:
                err = exc
            write(1, next(end))
            if err is not None:
                row(classify(err))
                err = None
                continue
            try:
                value = normalize(value, 0, [0])
                size = len(dumps(value, allow_nan=True))
            except MemoryError:
                row({"k": "resource", "e": "memory: result"})
                continue
            except (ValueError, RecursionError, TypeError):
                row({"k": "crash", "e": "unserializable"})
                continue
            if size > max_value or total + size > max_total:
                row({"k": "crash", "e": "output too large"})
                continue
            total += size
            row({"k": "ok", "v": value})

    return run


def _probe(kind, real, modules):
    """Host-requested self-report (tests only). Candidate source never selects this path."""
    if kind == "surface":
        import time
        report = {"version": sys.version.split()[0], "path": sys.path, "env": dict(os.environ), "missing": {}}
        for label, action in (("cwd", os.getcwd), ("root", lambda: sorted(os.listdir("/"))),
                              ("lib", lambda: sorted(os.listdir("/lib"))), ("time", time.time),
                              ("monotonic_ns", lambda: [time.monotonic_ns(), time.monotonic_ns()]),
                              ("urandom", lambda: os.urandom(8).hex()),
                              ("stat_lib", lambda: list(os.stat("/lib"))),
                              ("stat_zip", lambda: list(os.stat("/lib/python314.zip"))),
                              ("scandir_ino", lambda: [e.inode() for e in os.scandir("/lib")]),
                              ("sleep", lambda: time.sleep(0.01))):
            try:
                report[label] = action()
            except BaseException as exc:
                report[label] = "error: " + type(exc).__name__
        for name, exports in modules.items():
            absent = [attr for attr in exports if not hasattr(real[name], attr)]
            if absent:
                report["missing"][name] = absent
        for label, action in (("write_lib", lambda: open("/lib/x", "w")), ("read_etc", lambda: open("/etc/passwd")),
                              ("listdir_up", lambda: os.listdir("/lib/..")), ("socket", _socket)):
            try:
                action()
                report[label] = "allowed"
            except BaseException as exc:
                report[label] = type(exc).__name__
        return report
    if kind == "reach":
        return _reach(real, modules)
    return {"error": "unknown probe"}


def _socket():
    import socket
    socket.create_connection(("1.1.1.1", 53), timeout=1)


def _reach(real, modules):
    """Walk every public attribute reachable from the exported names (depth 4) and report modules,
    frames, code objects, loaders and import machinery: everything a gated candidate must not reach."""
    seen, found = set(), []
    frontier = [(name + "." + attr, getattr(real[name], attr))
                for name, exports in modules.items() for attr in exports if hasattr(real[name], attr)]
    for depth in range(4):
        nxt = []
        for path, obj in frontier:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            bad = (isinstance(obj, (types.ModuleType, types.FrameType, types.CodeType, types.TracebackType))
                   or obj is os or obj is sys or obj is builtins or obj is __import__
                   or hasattr(obj, "exec_module") or hasattr(obj, "load_module") or hasattr(obj, "find_spec"))
            if bad:
                found.append(path + ": " + type(obj).__name__)
                continue
            if len(seen) > 50000:
                break
            try:
                names = [n for n in dir(obj) if not n.startswith("_")]
            except BaseException:
                continue
            for attr in names:
                try:
                    nxt.append((path + "." + attr, getattr(obj, attr)))
                except BaseException:
                    pass
        frontier = nxt
    return {"visited": len(seen), "found": found}


if __name__ == "__main__":
    # The root frame is the candidate's outermost ancestor: clear its globals, then run. The closure
    # travels as an argument, so nothing in the module namespace survives.
    (lambda run: (globals().clear(), run is not None and run()))(_setup())
