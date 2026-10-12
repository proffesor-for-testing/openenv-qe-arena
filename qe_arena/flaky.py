"""qe_flaky: diagnose and repair a simulated flaky test. Harness, region gate, ground truth and grading.

module.py holds a corpus function, a suite of 3-5 tests (the "region") and a fixed harness
``run_suite(order, seed, t0)``. The sandbox has no clock, no ``random`` and a pinned hash seed, so flakiness is
simulated deterministically: each run builds a fresh ``env`` whose seeded streams (draw, frac), fake clock (now)
and unordered-collection order (pool) derive from (order, seed, t0), and module-level dicts carry state between
tests within a run. Generation lives in ``flaky_gen``.

Grading never runs the harness text the model can edit. It composes a grader-only module in which the harness
is defined before the region and captures every builtin it uses as a default argument, so nothing the region
defines can shadow it; the region only ever receives an ``env`` instance of a class built for that one run. The
region must also pass ``gate_region``: no harness names, no rebinding of builtins or of the function under test,
no attribute assignment.
"""

from __future__ import annotations

import ast
import builtins
import random

from . import sandbox
from .tasks import MODULE, REPLY_FORMAT, Task, _outputs

CAUSES = ("shared_state", "cache_leak", "shared_random", "clock_boundary", "unordered_iteration", "float_equality")
CAUSE_TEXT = {
    "shared_state": "assigns to an entry of a module-level dict that another test reads directly, so the outcome "
                    "depends on test order",
    "cache_leak": "a memoizing helper hands back a value stored by an earlier call because its key is too coarse",
    "shared_random": "the outcome depends on which value env.draw returns",
    "clock_boundary": "the outcome depends on env.now crossing a time boundary between two reads",
    "unordered_iteration": "the outcome depends on the order env.pool returns",
    "float_equality": "exact == on floating-point arithmetic whose rounding depends on the env.frac value",
}
FACILITY = {"shared_state": "state", "cache_leak": "store", "shared_random": "draw", "clock_boundary": "now",
            "unordered_iteration": "pool", "float_equality": "frac"}
MASK = {"shared_state": 1, "cache_leak": 2, "shared_random": 3, "float_equality": 4, "clock_boundary": 5,
        "unordered_iteration": 6}
SEED_SPLIT = 500_000  # shown runs use seeds below this, graded runs seeds at or above it
TICK = 250
MAX_GRADED_BYTES = 5600  # leaves room under the sandbox's 6000-byte cap for a repair to grow the test
HARNESS_HEAD = "from collections import namedtuple\n"
HARNESS_NAMES = frozenset({"SUITE", "BOXES", "SNAPSHOT", "RUN", "make_runner", "run_suite", "namedtuple", "probe",
                           "sweep", "permutations", "Env"})
BUILTIN_NAMES = frozenset(dir(builtins))

CORE = '''def make_runner(namedtuple=namedtuple, list=list, range=range, len=len, zip=zip, Exception=Exception{gargs}):
    """Builds run(). Every run gets a fresh env: its own Env class, seeded streams and clock."""
    tick = 250

    def run(suite, boxes, snapshot, order, seed, t0{rargs}):
        for box, saved in zip(boxes, snapshot):
            box.clear()
            box.update(saved)
        cell = [seed % 2147483646 + 1, (seed * 7919 + 17) % 2147483646 + 1, (seed * 104729 + 3) % 2147483646 + 1, t0]

        def step(i):
            cell[i] = cell[i] * 48271 % 2147483647
            return cell[i]

        def draw(n):
            """An int in range(n) from the run's seeded stream, which every test in the run shares."""
            if stub and stub[0] == "draw":  #G
                return stub[1] % n  #G
            return step(0) % n

        def frac():
            """A number in (0, 1) from a second seeded stream."""
            if stub and stub[0] == "frac":  #G
                return stub[1]  #G
            return step(1) / 2147483647

        def now():
            """Fake clock in ms: t0 at the first read of the run, then 250 more on every read."""
            cell[3] += tick
            return cell[3] - tick

        def pool(items):
            """items in the iteration order of an unordered collection, which changes from run to run."""
            items = list(items)
            if stub and stub[0] == "pool":  #G
                pick = list(permutations(range(len(items))))  #G
                return [items[i] for i in pick[stub[1] % len(pick)]]  #G
            for i in range(len(items) - 1, 0, -1):
                j = step(2) % (i + 1)
                items[i], items[j] = items[j], items[i]
            return items

        env = namedtuple("Env", "draw frac now pool")(draw, frac, now, pool)
        passed = {{}}
        for index in order:
            for k in reset:  #G
                boxes[k].clear()  #G
                boxes[k].update(snapshot[k])  #G
            if mask in (3, 4, 6):  #G
                cell[{{3: 0, 4: 1, 6: 2}}[mask]] = 1  #G
            if mask == 5:  #G
                cell[3] = 0  #G
            name, test = suite[index]
            try:
                test(env)
                passed[name] = True
            except Exception:
                passed[name] = False
        return passed

    return run


RUN = make_runner()
'''
RUN_SUITE = '''def run_suite(order, seed, t0):
    """Run SUITE[i] for each i in order, as one CI run does. Returns {test name: passed}."""
    return RUN(SUITE, BOXES, SNAPSHOT, order, seed, t0)
'''
GRADER_TAIL = '''

def probe(order, seed, t0, mask, reset):
    return RUN(SUITE, BOXES, SNAPSHOT, order, seed, t0, mask, reset)


def sweep(name, kind, value):
    one = [item for item in SUITE if item[0] == name]
    return RUN(one, BOXES, SNAPSHOT, [0], 1, value if kind == "now" else 0, 0, (), [kind, value])
'''


def _core(grader: bool) -> str:
    text = CORE.format(gargs=", permutations=permutations" if grader else "",
                       rargs=", mask=0, reset=(), stub=None" if grader else "")
    lines = []
    for line in text.split("\n"):
        if line.endswith("  #G"):
            if grader:
                lines.append(line[:-4])
        elif not (grader and line.strip().startswith('"""')):
            lines.append(line)
    return "\n".join(lines)


def _registry(names: list[str], boxes: list[str]) -> str:
    return (f"SUITE = [{', '.join(f'({n!r}, {n})' for n in names)}]\nBOXES = [{', '.join(boxes)}]\n"
            "SNAPSHOT = [dict(box) for box in BOXES]\n")


def shown_harness(names: list[str], boxes: list[str]) -> str:
    return f"{HARNESS_HEAD}\n{_registry(names, boxes)}\n\n{_core(False)}\n\n{RUN_SUITE}"


def graded_source(prefix: str, region: str, names: list[str], boxes: list[str]) -> str:
    """The module the grader runs: harness first, so it is bound before any region code executes."""
    head = f"{HARNESS_HEAD}from itertools import permutations\n\n\n{_core(True)}\n\n"
    return prefix + head + region + _registry(names, boxes) + "\n\n" + RUN_SUITE + GRADER_TAIL


def top_names(source: str) -> set[str]:
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def gate_region(region: str, protected: set[str]) -> list[str]:
    """Reasons the tests region may not run: harness names, rebinding builtins or ``protected`` names (the
    function under test and its helpers), or assigning an attribute. Empty means it passed."""
    try:
        tree = ast.parse(region)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return ["the tests do not parse"]
    reserved = BUILTIN_NAMES | protected | HARNESS_NAMES
    problems = []

    def bound(name: str | None) -> None:
        if name and name in reserved:
            problems.append(f"the tests may not bind {name!r}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in HARNESS_NAMES:
                problems.append(f"{node.id!r} belongs to the harness")
            elif not isinstance(node.ctx, ast.Load):
                bound(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound(node.name)
        elif isinstance(node, ast.arg):
            bound(node.arg)
        elif isinstance(node, ast.alias):
            bound((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            bound(node.name)
        elif isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
            problems.append(f"the tests may not assign attribute {node.attr!r}")
    return list(dict.fromkeys(problems))


def run(source: str, calls: list[list], entry: str = "run_suite") -> list[dict] | None:
    results = []
    for start in range(0, len(calls), 200):
        chunk = _outputs(source, entry, calls[start:start + 200])
        if not all(r.kind == "ok" and isinstance(r.value, dict) for r in chunk):
            return None
        results.extend(r.value for r in chunk)
    return results


def combos(rng: random.Random, n_tests: int, count: int, shown: bool) -> list[list]:
    lo, hi = (0, SEED_SPLIT) if shown else (SEED_SPLIT, 2 * SEED_SPLIT)
    return [[rng.sample(range(n_tests), n_tests), rng.randrange(lo, hi), rng.randrange(10**7)] for _ in range(count)]


def sweep_calls(rng: random.Random) -> list[list]:
    """Adversarial facility values for the repaired test alone: every draw residue 0..7 (and large ones), clocks
    just below powers of ten and calendar periods (and random ones), a frac grid, and every pool order."""
    draws = list(range(8)) + [rng.randrange(8, 10**9) for _ in range(8)]
    clocks = [10**k - d for k in range(1, 14) for d in (1, 125, 249) if 10**k > d]
    clocks += [p * m - d for p in (60_000, 3_600_000, 86_400_000) for m in (1, 7) for d in (1, 249)]
    clocks += [rng.randrange(10**13) for _ in range(48)]
    fracs = [i / 64 for i in range(1, 64)] + [i / 10 for i in range(1, 10)] + [i / 97 for i in range(1, 97, 3)]
    fracs += [rng.randrange(1, 2147483647) / 2147483647 for _ in range(48)]
    return ([["draw", v] for v in draws] + [["now", v] for v in clocks] + [["frac", v] for v in fracs]
            + [["pool", v] for v in range(6)])


def attribute(source: str, graded: list[list], victim: str, idx: dict) -> tuple[list[str], list[dict]] | None:
    """Causes whose facility, reset before every test, makes the victim's outcome constant (plus mask-0 runs)."""
    calls = []
    for mask in range(7):
        reset = idx["state"] if mask == 1 else idx["store"] if mask == 2 else []
        calls += [c + [mask, reset] for c in graded]
    outcomes = run(source, calls, entry="probe")
    if outcomes is None:
        return None
    k = len(graded)
    stable = [m for m in range(1, 7) if len({o.get(victim) for o in outcomes[m * k:(m + 1) * k]}) == 1]
    return [c for c in CAUSES if MASK[c] in stable], outcomes[:k]


def victim_block(source: str, name: str) -> str | None:
    """The text of the top-level test function ``name`` in ``source``, or None."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return ast.get_source_segment(source, found[0]) if len(found) == 1 else None


def spliced_region(task: Task, source: str) -> str | None:
    """The original tests with only the flaky test's function taken from ``source`` (the graded repair)."""
    p = task.private
    edited = victim_block(source, p["victim"])
    if edited is None:
        return None
    original = victim_block(p["region"], p["victim"])
    return p["region"].replace(original, edited, 1)


def repair_ok(task: Task, region: str, mutant: str | None = None) -> bool:
    """The suite passes on every graded run, the repaired test passes every sweep value alone, and it fails on
    every graded run when the function under test is the reference mutant."""
    p = task.private
    if gate_region(region, set(p["protected"])):
        return False
    source = graded_source(p["prefix"], region, p["tests"], p["boxes"])
    if len(source.encode()) > 6000:
        return False
    outcomes = run(source, p["graded"])
    if outcomes is None or not all(sorted(o) == sorted(p["tests"]) and all(v is True for v in o.values())
                                   for o in outcomes):
        return False
    swept = run(source, [[p["victim"]] + s for s in p["sweep"]], entry="sweep")
    if swept is None or not all(o.get(p["victim"]) is True for o in swept):
        return False
    mutant = mutant or p["mutants"][0]["source"]
    killed = run(graded_source(mutant.rstrip("\n") + "\n\n\n", region, p["tests"], p["boxes"]), p["graded"])
    return killed is not None and all(o.get(p["victim"]) is False for o in killed)


def _structure(task: Task, source: str) -> str | None:
    p = task.private
    if not (source.startswith(p["prefix"]) and source.endswith(p["harness"])
            and len(source) >= len(p["prefix"]) + len(p["harness"])):
        return None
    return source[len(p["prefix"]):len(source) - len(p["harness"])]


def run_shown(task: Task, source: str) -> str:
    """Run the working copy's tests on the shown runs (the model's run_tests)."""
    p = task.private
    region = _structure(task, source)
    if region is None:
        return "the code under test and the harness must stay unchanged; only the tests may be edited"
    problems = gate_region(region, set(p["protected"]))
    graded = graded_source(p["prefix"], region, p["tests"], p["boxes"])
    problems = problems or sandbox.static_gate(graded)
    if problems:
        return "REJECTED: " + "; ".join(problems)[:400]
    results = _outputs(graded, "run_suite", p["shown"])
    lines = []
    for (order, seed, t0), result in zip(p["shown"], results):
        head = f"run_suite({order}, {seed}, {t0})"
        if result.kind != "ok" or not isinstance(result.value, dict):
            lines.append(f"{head}: {result.kind} {result.error[:120]}")
            continue
        failed = [str(n)[:40] for n, ok in result.value.items() if ok is not True]
        lines.append(f"{head}: {len(result.value) - len(failed)}/{len(result.value)} passed"
                     + (f"; failed: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)


def _answer(answer) -> tuple[str | None, str | None] | None:
    if not isinstance(answer, dict) or "test" not in answer or set(answer) - {"test", "cause"}:
        return None
    test, cause = answer["test"], answer.get("cause")
    if not (test is None or type(test) is str) or not (cause is None or type(cause) is str):
        return None
    return test, cause


def grade(task: Task, answer, source: str) -> float:
    p = task.private
    parsed = _answer(answer)
    if parsed is None:
        return 0.0
    test, cause = parsed
    if p["victim"] is None:  # clean control: only "no flaky test" is right; d3 edits are discarded
        return 1.0 if test is None else 0.0
    if test != p["victim"]:
        return 0.0
    if task.difficulty == 1:
        return 1.0
    if task.difficulty == 2:
        return 1.0 if cause == p["cause"] else 0.5
    region = spliced_region(task, source)
    return 1.0 if region is not None and repair_ok(task, region) else 0.5


def prompt(entry: str, level: int, shown: list[list]) -> str:
    runs = "; ".join(f"run_suite({o}, {s}, {t})" for o, s, t in shown)
    lines = [
        f"{MODULE} holds `{entry}` (the code under test), a test suite that checks it, and a fixed harness "
        "`run_suite(order, seed, t0)`. CI runs the suite many times with different orders, seeds and start times. "
        "Every test gets `env`: env.draw(n) and env.frac() read seeded pseudo-random streams shared by the whole "
        "run, env.now() is a fake millisecond clock, and env.pool(items) returns items in an order that changes "
        "between runs. Module-level dicts are reset at the start of each run, not between tests.",
        "A test is flaky when its outcome changes between runs although the code under test does not. At most one "
        "test in this suite is flaky, and possibly none is. Some flaky tests fail only rarely.",
        f"Every test passed on these runs: {runs}.",
    ]
    if level == 1:
        lines.append("Name the flaky test, or answer null when no test is flaky.")
        answer = '{"op": "submit", "answer": {"test": "test_name"}}  or  {"op": "submit", "answer": {"test": null}}'
    elif level == 2:
        lines.append("Name the flaky test and its root cause, or answer null when no test is flaky. Causes:")
        lines.extend(f"- {c}: {CAUSE_TEXT[c]}" for c in CAUSES)
        lines.append("The right test earns half the reward; the right test and cause earn all of it.")
        answer = ('{"op": "submit", "answer": {"test": "test_name", "cause": "shared_state"}}  or  '
                  '{"op": "submit", "answer": {"test": null}}')
    else:
        lines.append(
            "Repair the flaky test so it passes on every run and still checks "
            f"`{entry}`, then submit its name, or null when no test is flaky. Only the flaky test's own function is "
            "graded: edit inside it, because every other edit is discarded. Test code may not use the harness's "
            "names, rebind built-in names, or assign attributes. Naming the flaky test earns half the reward. Full "
            "reward needs a suite that passes on every run, under any value env can produce, and a repaired test "
            f"that still fails when `{entry}` is wrong, so deleting or weakening its checks earns no more than half.")
        answer = '{"op": "submit", "answer": {"test": "test_name"}}  or  {"op": "submit", "answer": {"test": null}}'
    lines += ["", REPLY_FORMAT, 'First action: {"op": "read", "path": "*"}',
              'Run the suite on the runs above (at most 3 times): {"op": "run_tests"}']
    if level == 3:
        lines.append('Edit by exact text replacement (old must occur exactly once): '
                     '{"op": "edit", "path": "module.py", "old": "    assert x == 1", "new": "    assert y == 1"}')
    lines.append(f"Final action: {answer}")
    return "\n".join(lines)


def make_flaky_task(task_id: str, level: int, seed: int, rng: random.Random, pool: tuple[str, ...]) -> Task:
    from .flaky_gen import build_task  # noqa: PLC0415 - flaky_gen imports this module

    return build_task(task_id, level, seed, rng, pool)
