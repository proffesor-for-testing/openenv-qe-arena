"""qe_flaky: diagnose and repair a simulated flaky test.

module.py holds a corpus function, a suite of 3-5 tests that check it, and a fixed harness
``run_suite(order, seed, t0)``. The sandbox has no clock, no ``random`` and a pinned hash seed, so
flakiness is simulated deterministically: every test receives an ``env`` whose pseudo-random streams,
fake clock and unordered-collection order derive from (order, seed, t0), and module-level dicts carry
state between tests within one run. One test carries one injected root cause; the other tests touch the
same facilities safely, often in a form that looks like the faulty one, so no surface feature names the
flaky test. About 15% of tasks are clean controls in which no test is flaky.

Ground truth comes from execution, never from the generator's label. The suite runs under K hidden
(order, seed, t0) combinations; exactly one test must both pass and fail, every other test must always
pass, and the cause is the one facility whose per-test reset (a grader-only harness variant) makes that
test's outcome constant. A task where any of this disagrees with what was injected is regenerated.
"""

from __future__ import annotations

import ast
import json
import random
from collections import Counter
from dataclasses import dataclass, field

from . import corpus, mutation, sandbox
from .tasks import MODULE, REPLY_FORMAT, Task, _outputs

CAUSES = ("shared_state", "cache_leak", "shared_random", "clock_boundary", "unordered_iteration", "float_equality")
CAUSE_TEXT = {
    "shared_state": "a test changes a module-level dict that another test reads, so the outcome depends on test order",
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
CLEAN_RATE = 0.15
K_GRADE = 24
N_SHOWN = 2
N_TESTS = {1: (3, 4), 2: (4, 5), 3: (4, 5)}
P_MIX = {1: 0.2, 2: 0.5, 3: 0.4}    # a test carries a second, safe check on another facility
P_SAME = {1: 0.4, 2: 0.8, 3: 0.8}   # a decoy uses the flaky test's facility
P_LOOK = {1: 0.2, 2: 0.6, 3: 0.5}   # a safe check is written to look like the faulty one
MAX_FUNCTION_BYTES = 1800
MAX_LITERAL = 48
SEED_SPLIT = 500_000  # shown runs use seeds below this, graded runs seeds at or above it
TELLTALE = ("flak", "random", "cache", "memo", "leak", "shared", "unorder", "float", "clock", "boundary",
            "order", "victim", "pollut", "culprit", "race", "time")

TEST_WORDS = ("amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
              "kestrel", "lagoon", "marble", "nectar", "onyx", "prairie", "quartz", "raven", "sierra", "tundra",
              "umber", "violet", "willow", "zephyr", "alder", "birch", "comet", "dune", "fern", "glacier")
STATE_NAMES = ("DEFAULTS", "SETTINGS", "FIXTURE", "PARAMS", "PROFILE", "BASELINE")
KEY_NAMES = ("input", "case", "args", "sample", "probe", "subject", "item", "seed_case")
STORE_NAMES = ("STORE", "TABLE", "LEDGER", "SLOTS", "INDEX", "BANK")
HELPER_NAMES = ("lookup", "fetch", "resolve", "obtain", "evaluate", "answer_for")
FLOAT_BAD = ("(w + 0.1) - 0.1 == w", "w * 0.1 * 10 == w", "(w + 0.5) - 0.5 == w", "(w + 0.2) + 0.1 == w + 0.3",
             "w * 3 / 3 == w", "w / 10 * 10 == w")
FLOAT_LOOK = ("w * 4 / 4 == w", "w + w + w == 3 * w", "w / 2 * 2 == w", "w * 2 - w == w")
FLOAT_SAFE = ("abs((w + 0.1) - 0.1 - w) < 1e-9", "0 < w < 1")

HARNESS = '''from collections import namedtuple

Env = namedtuple("Env", "draw frac now pool")
SUITE = [{suite}]
BOXES = [{boxes}]
SNAPSHOT = [dict(box) for box in BOXES]
TICK = 250


def run_suite(order, seed, t0):
    """Run SUITE[i] for each i in order, as one CI run does. Returns {{test name: passed}}."""
    for box, saved in zip(BOXES, SNAPSHOT):
        box.clear()
        box.update(saved)
    cell = [seed % 2147483646 + 1, (seed * 7919 + 17) % 2147483646 + 1, (seed * 104729 + 3) % 2147483646 + 1, t0]

    def step(i):
        cell[i] = cell[i] * 48271 % 2147483647
        return cell[i]

    def draw(n):
        """An int in range(n) from the run's seeded stream, which every test in the run shares."""
        return step(0) % n

    def frac():
        """A number in (0, 1) from a second seeded stream."""
        return step(1) / 2147483647

    def now():
        """Fake clock in ms: t0 at the first read of the run, then TICK more on every read."""
        cell[3] += TICK
        return cell[3] - TICK

    def pool(items):
        """items in the iteration order of an unordered collection, which changes from run to run."""
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = step(2) % (i + 1)
            items[i], items[j] = items[j], items[i]
        return items

    env = Env(draw, frac, now, pool)
    passed = {{}}
    for index in order:
        name, test = SUITE[index]
        try:
            test(env)
            passed[name] = True
        except Exception:
            passed[name] = False
    return passed
'''
# Grader-only: reset one facility before every test. Never shown; used to attribute the cause by execution.
PROBE_HOOK = '''        for k, (box, saved) in enumerate(zip(BOXES, SNAPSHOT)):
            if (mask == 1 and k in STATE_IDX) or (mask == 2 and k in MEMO_IDX):
                box.clear()
                box.update(saved)
        if mask in (3, 4, 6):
            cell[{3: 0, 4: 1, 6: 2}[mask]] = 1
        if mask == 5:
            cell[3] = 0
'''


@dataclass
class _Check:
    """One check inside a test: its lines, the args it calls the function with, and the repaired lines."""
    lines: list[str]
    args: list[list]
    repair: list[str] | None = None


@dataclass
class _Plan:
    entry: str
    rows: list[tuple[list, object]]
    rng: random.Random
    used: set = field(default_factory=set)
    state: dict = field(default_factory=dict)   # key -> initial args
    store_users: list = field(default_factory=list)
    zones: dict | None = None   # d3: {"kill": args the mutant returns a different value for, "spare": same value}
    zone: str = ""              # which of those zones take() draws from

    def allowed(self, row) -> bool:
        return self.zones is None or not self.zone or json.dumps(row[0]) in self.zones[self.zone]

    def call(self, args: list) -> str:
        return f"{self.entry}({', '.join(repr(a) for a in args)})"

    def take(self, count: int, same_output: bool = False, ok=None) -> list[tuple[list, object]] | None:
        """Unused rows: ``count`` with pairwise different outputs, or two with equal outputs."""
        free = [r for r in self.rows if json.dumps(r[0]) not in self.used and self.allowed(r) and (ok is None or ok(r))]
        self.rng.shuffle(free)
        if same_output:
            for i, a in enumerate(free):
                for b in free[i + 1:]:
                    if a[1] == b[1] and type(a[1]) is type(b[1]):
                        return self._mark([a, b])
            return None
        chosen = []
        for row in free:
            if all(row[1] != c[1] for c in chosen):
                chosen.append(row)
                if len(chosen) == count:
                    return self._mark(chosen)
        return None

    def _mark(self, rows):
        self.used.update(json.dumps(r[0]) for r in rows)
        return rows


def _key_repr(args: list) -> str:
    return repr(tuple(args))


def _snippet(p: _Plan, facility: str, mode: str, names: dict) -> _Check | None:
    """mode: culprit (the injected cause), twin (its safe twin for clean controls), look or safe (decoys)."""
    E, rng = p.entry, p.rng
    if mode == "twin":  # state and store keep the reader; their twin differs in the writer or the key
        mode = "culprit" if facility in ("state", "store") else "look"
    if facility == "draw":
        if mode == "culprit":
            (a, oa), (b, _) = p.take(2) or ((None, None), (None, None))
            if a is None:
                return None
            if rng.random() < 0.5:
                pair = [a, b] if rng.random() < 0.5 else [b, a]
                first = f"args = [{', '.join(repr(x) for x in pair)}][env.draw(2)]"
            else:
                first = f"args = {a!r} if env.draw({rng.choice((2, 3, 4))}) else {b!r}"
            return _Check([first, f"assert {E}(*args) == {oa!r}"], [a, b], [f"assert {p.call(a)} == {oa!r}"])
        if mode == "look" and (same := p.take(2, same_output=True)):
            (a, oa), (b, _) = same
            return _Check([f"args = [{a!r}, {b!r}][env.draw(2)]", f"assert {E}(*args) == {oa!r}"], [a, b])
        if rng.random() < 0.5 and (rows := p.take(2)):
            (a, oa), (b, ob) = rows
            return _Check([f"args, want = [({a!r}, {oa!r}), ({b!r}, {ob!r})][env.draw(2)]",
                           f"assert {E}(*args) == want"], [a, b])
        (a, oa), = p.take(1) or ((None, None),)
        if a is None:
            return None
        return _Check(["count = 1 + env.draw(3)", f"assert [{p.call(a)}] * count == [{oa!r}] * count"], [a])
    if facility == "frac":
        (a, oa), = p.take(1) or ((None, None),)
        if a is None:
            return None
        expr = rng.choice({"culprit": FLOAT_BAD, "look": FLOAT_LOOK, "safe": FLOAT_SAFE}[mode])
        lines = ["w = env.frac()", f"assert {p.call(a)} == {oa!r}", f"assert {expr}"]
        return _Check(lines, [a], [f"assert {p.call(a)} == {oa!r}"] if mode == "culprit" else None)
    if facility == "now":
        (a, oa), = p.take(1) or ((None, None),)
        if a is None:
            return None
        bucket = rng.choice((500, 1000))
        if mode == "culprit":
            return _Check([f"stamp = env.now() // {bucket}", f"got = {p.call(a)}",
                           f"assert got == {oa!r} and env.now() // {bucket} == stamp"], [a],
                          [f"got = {p.call(a)}", f"assert got == {oa!r}"])
        if mode == "look":
            return _Check([f"stamp = env.now() // {bucket}", f"got = {p.call(a)}",
                           f"assert got == {oa!r} and env.now() // {bucket} >= stamp"], [a])
        return _Check(["start = env.now()", f"got = {p.call(a)}",
                       f"assert got == {oa!r} and env.now() - start < {bucket}"], [a])
    if facility == "pool":
        if mode == "look" and (same := p.take(2, same_output=True)):
            (a, oa), (b, _) = same
            if rng.random() < 0.5:
                return _Check([f"first = env.pool([{a!r}, {b!r}])[0]", f"assert {E}(*first) == {oa!r}"], [a, b])
            return _Check([f"got = [{E}(*a) for a in env.pool([{a!r}, {b!r}])]", f"assert got == [{oa!r}, {oa!r}]"],
                          [a, b])
        rows = p.take(2)
        if rows is None:
            return None
        (a, oa), (b, ob) = rows
        if mode == "culprit":
            if rng.random() < 0.5:
                return _Check([f"first = env.pool([{a!r}, {b!r}])[0]", f"assert {E}(*first) == {oa!r}"], [a, b],
                              [f"assert {p.call(a)} == {oa!r}"])
            return _Check([f"got = [{E}(*a) for a in env.pool([{a!r}, {b!r}])]", f"assert got == [{oa!r}, {ob!r}]"],
                          [a, b], [f"got = [{E}(*a) for a in [{a!r}, {b!r}]]", f"assert got == [{oa!r}, {ob!r}]"])
        return _Check([f"got = [{E}(*a) for a in env.pool([{a!r}, {b!r}])]",
                       f"assert len(got) == 2 and {oa!r} in got and {ob!r} in got"], [a, b])
    if facility == "state":
        D = names["state"]
        (a, oa), = p.take(1) or ((None, None),)
        if a is None:
            return None
        if mode == "culprit":  # the reader; its writer is added by the caller
            key = names["state_key"]
            p.state[key] = a
            return _Check([f"assert {E}(*{D}[{key!r}]) == {oa!r}"], [a], [f"assert {p.call(a)} == {oa!r}"])
        key = rng.choice([k for k in KEY_NAMES if k not in p.state and k != names["state_key"]])
        if mode == "look":
            return _Check([f"{D}[{key!r}] = {a!r}", f"assert {E}(*{D}[{key!r}]) == {oa!r}"], [a])
        p.state[key] = a
        return _Check([f"assert {E}(*{D}[{key!r}]) == {oa!r}"], [a])
    if facility == "store":
        H = names["helper"]
        if mode == "culprit":
            a, oa = names["store_pair"][0]
            p.store_users.append(a)
            return _Check([f"assert {H}({', '.join(repr(x) for x in a)}) == {oa!r}"], [a],
                          [f"assert {p.call(a)} == {oa!r}"])
        prefix = names.get("store_prefix", 64)
        clash = {_key_repr(u)[:prefix] for u in p.store_users} | {_key_repr(r[0])[:prefix] for r in
                                                                   names.get("store_pair", ())}
        (a, oa), = p.take(1, ok=lambda r: _key_repr(r[0])[:prefix] not in clash) or ((None, None),)
        if a is None:
            return None
        p.store_users.append(a)
        return _Check([f"assert {H}({', '.join(repr(x) for x in a)}) == {oa!r}"], [a])
    raise ValueError(facility)


def _writer(p: _Plan, cause: str, clean: bool, names: dict) -> _Check | None:
    E = p.entry
    if cause == "shared_state":
        D, key = names["state"], names["state_key"]
        (b, ob), = p.take(1) or ((None, None),)
        if b is None:
            return None
        if clean:
            return _Check([f"saved = {D}[{key!r}]", f"{D}[{key!r}] = {b!r}", "try:",
                           f"    assert {E}(*{D}[{key!r}]) == {ob!r}", "finally:", f"    {D}[{key!r}] = saved"], [b])
        return _Check([f"{D}[{key!r}] = {b!r}", f"assert {E}(*{D}[{key!r}]) == {ob!r}"], [b])
    b, _ = names["store_pair"][1]
    p.store_users.append(b)
    return _Check([f"assert {names['helper']}({', '.join(repr(x) for x in b)}) is not None"], [b])


def _store_pair(p: _Plan) -> tuple[tuple, tuple, int] | None:
    """Two rows whose call keys share a prefix of at least 4 characters but whose outputs differ."""
    free = [r for r in p.rows if json.dumps(r[0]) not in p.used and r[1] is not None]
    p.rng.shuffle(free)
    zone = p.zone
    for i, a in enumerate(free):
        for b in free[i + 1:]:
            if p.zones is not None:  # the reader's input must be killed, the writer's spared
                p.zone = "kill"
                if not p.allowed(a):
                    a, b = b, a
                ok = p.allowed(a)
                p.zone = "spare"
                ok = ok and p.allowed(b)
                p.zone = zone
                if not ok:
                    continue
            ka, kb = _key_repr(a[0]), _key_repr(b[0])
            common = next((n for n in range(min(len(ka), len(kb))) if ka[n] != kb[n]), None)
            if common is not None and common >= 4 and a[1] != b[1]:
                p._mark([a, b])
                return a, b, common
    return None


def _key_expr(p: _Plan, cause: str, clean: bool, names: dict) -> str | None:
    if cause == "cache_leak" and not clean:
        return f"repr(args)[:{names['store_prefix']}]"
    keys = [_key_repr(a) for a in p.store_users]
    if p.rng.random() < 0.5:
        return "repr(args)"
    for n in range(4, 40):  # a coarse-looking key that still tells every call in this suite apart
        if len({k[:n] for k in keys}) == len(keys):
            return f"repr(args)[:{n}]"
    return "repr(args)"


def _eligible(pool: tuple[str, ...]) -> list[str]:
    names = []
    for name in sorted(pool):
        item = corpus.ENTRIES[name]
        if len(item.function_source) <= MAX_FUNCTION_BYTES and "flak" not in item.function_source.lower():
            names.append(name)
    return names


def _rows(name: str, source: str, entry: str, rng: random.Random) -> list[tuple[list, object]]:
    inputs = corpus.gen_inputs(name, rng.randrange(2**31), 64)
    results = _outputs(source, entry, inputs)
    rows, seen, size = [], set(), 0
    for args, result in zip(inputs, results):
        text = json.dumps(args)
        out = json.dumps(result.value)
        if result.kind != "ok" or text in seen or len(text) > MAX_LITERAL or len(out) > MAX_LITERAL:
            continue
        if any(word in (text + out).lower() for word in TELLTALE):
            continue
        seen.add(text)
        size += len(repr((args, result.value))) + 2
        if size > 3600:  # the check below must stay under the sandbox's source cap
            break
        rows.append((args, result.value))
    if not rows:
        return []
    # The test compares Python values; keep only rows whose literal equals what the function returns there.
    check = source + f"\n\nROWS = {[(a, o) for a, o in rows]!r}\n\n\ndef same(i):\n    a, o = ROWS[i]\n" \
        f"    return {entry}(*a) == o\n"
    verdicts = _outputs(check, "same", [[i] for i in range(len(rows))])
    return [row for row, v in zip(rows, verdicts) if v.kind == "ok" and v.value is True]


def _combos(rng: random.Random, n_tests: int, count: int, shown: bool) -> list[list]:
    lo, hi = (0, SEED_SPLIT) if shown else (SEED_SPLIT, 2 * SEED_SPLIT)
    return [[rng.sample(range(n_tests), n_tests), rng.randrange(lo, hi), rng.randrange(10**7)] for _ in range(count)]


def _run(source: str, combos: list[list], entry: str = "run_suite") -> list[dict] | None:
    results = _outputs(source, entry, combos)
    if not all(r.kind == "ok" and isinstance(r.value, dict) for r in results):
        return None
    return [r.value for r in results]


def _render(p: _Plan, tests: list[tuple[str, list[_Check]]], names: dict, key_expr: str | None,
            repaired: str | None = None) -> tuple[str, list[str]]:
    """The tests region (containers, helper, tests) and the names of the module-level dicts."""
    parts, boxes = [], []
    if p.state:
        parts.append(f"{names['state']} = {dict(sorted(p.state.items()))!r}")
        boxes.append(names["state"])
    if key_expr:
        S, H = names["store"], names["helper"]
        parts.append(f"{S} = {{}}\n\n\ndef {H}(*args):\n    key = {key_expr}\n    if key not in {S}:\n"
                     f"        {S}[key] = {p.entry}(*args)\n    return {S}[key]")
        boxes.append(S)
    for name, checks in tests:
        body = []
        for check in checks:
            lines = check.repair if (repaired == name and check.repair) else check.lines
            body.extend("    " + line for line in lines)
        parts.append(f"def {name}(env):\n" + "\n".join(body))
    return "\n\n\n".join(parts) + "\n\n\n", boxes


def _harness(names: list[str], boxes: list[str]) -> str:
    return HARNESS.format(suite=", ".join(f"({n!r}, {n})" for n in names), boxes=", ".join(boxes))


def _probe(harness: str, boxes: list[str], names: dict) -> str:
    state_idx = [i for i, b in enumerate(boxes) if b == names["state"]]
    memo_idx = [i for i, b in enumerate(boxes) if b == names["store"]]
    out = harness.replace("def run_suite(order, seed, t0):", "def probe(order, seed, t0, mask):")
    out = out.replace("        name, test = SUITE[index]\n", PROBE_HOOK + "        name, test = SUITE[index]\n")
    return f"STATE_IDX = {state_idx}\nMEMO_IDX = {memo_idx}\n" + out


def _varying(outcomes: list[dict], names: list[str]) -> tuple[list[str], list[str]]:
    """(tests that both pass and fail, tests that never pass in every run)."""
    vary, broken = [], []
    for n in names:
        seen = {o.get(n) for o in outcomes}
        if seen == {True, False}:
            vary.append(n)
        elif seen != {True}:
            broken.append(n)
    return vary, broken


def _attribute(probe: str, combos: list[list], victim: str) -> tuple[list[str], list[dict]] | None:
    """Causes whose facility, reset before every test, makes the victim's outcome constant."""
    calls = [c + [mask] for mask in range(7) for c in combos]
    outcomes = _run(probe, calls, entry="probe")
    if outcomes is None:
        return None
    k = len(combos)
    stable = [m for m in range(1, 7) if len({o.get(victim) for o in outcomes[m * k:(m + 1) * k]}) == 1]
    return [c for c in CAUSES if MASK[c] in stable], outcomes[:k]


def _mutant(function: str, entry: str, rows: list, rng: random.Random) -> tuple[str, dict] | None:
    """A reference mutant for d3 and the inputs it kills. The culprit check draws only killed inputs and
    every other test only spared ones, so the repaired test is the one check in the suite that notices it.
    A kill is a different returned value, never a raise: a raise would fail a test that asserts nothing."""
    calls = [a for a, _ in rows]
    reference = _outputs(function, entry, calls)
    sites = mutation.enumerate_sites(function)
    rng.shuffle(sites)
    for site in sites[:12]:
        mutant = mutation.apply(function, [site])
        if mutant.split("\n", 1)[-1] in function:  # never a copy of the shown body
            continue
        outputs = _outputs(mutant, entry, calls)
        if not all(o.kind in ("ok", "exc") for o in outputs):
            continue
        zones = {"kill": set(), "spare": set()}
        for args, r, o in zip(calls, reference, outputs):
            if sandbox.compare(r, o):
                zones["spare"].add(json.dumps(args))
            elif o.kind == "ok":
                zones["kill"].add(json.dumps(args))
        if len(zones["kill"]) >= 3 and len(zones["spare"]) >= 8:
            return mutant, zones
    return None


def _prompt(entry: str, level: int, shown: list[list]) -> str:
    runs = "; ".join(f"run_suite({o}, {s}, {t})" for o, s, t in shown)
    lines = [
        f"{MODULE} holds `{entry}` (the code under test), a test suite that checks it, and a fixed harness "
        "`run_suite(order, seed, t0)`. CI runs the suite many times with different orders, seeds and start times. "
        "Every test gets `env`: env.draw(n) and env.frac() read seeded pseudo-random streams shared by the whole "
        "run, env.now() is a fake millisecond clock, and env.pool(items) returns items in an order that changes "
        "between runs. Module-level dicts are reset at the start of each run, not between tests.",
        "A test is flaky when its outcome changes between runs although the code under test does not. At most one "
        "test in this suite is flaky, and possibly none is.",
        f"Every test passed on these runs: {runs}.",
    ]
    if level == 1:
        lines.append('Name the flaky test, or answer null when no test is flaky.')
        answer = '{"op": "submit", "answer": {"test": "test_name"}}  or  {"op": "submit", "answer": {"test": null}}'
    elif level == 2:
        lines.append("Name the flaky test and its root cause, or answer null when no test is flaky. Causes:")
        lines.extend(f"- {c}: {CAUSE_TEXT[c]}" for c in CAUSES)
        lines.append("The right test earns half the reward; the right test and cause earn all of it.")
        answer = ('{"op": "submit", "answer": {"test": "test_name", "cause": "shared_state"}}  or  '
                  '{"op": "submit", "answer": {"test": null}}')
    else:
        lines.append(
            f"Repair the flaky test with edits so it passes on every run and still checks `{entry}`. Edit only the "
            f"tests: `{entry}` and everything from `from collections import namedtuple` down must stay unchanged. "
            "Then submit the name of the test you repaired, or null when no test is flaky. Naming the flaky test "
            f"earns half the reward. Full reward needs a suite that passes on every run and a repaired test that "
            f"still fails when `{entry}` is wrong, so deleting or weakening its checks earns no more than half.")
        answer = '{"op": "submit", "answer": {"test": "test_name"}}  or  {"op": "submit", "answer": {"test": null}}'
    lines += ["", REPLY_FORMAT, 'First action: {"op": "read", "path": "*"}',
              'Run the suite on the runs above (at most 3 times): {"op": "run_tests"}']
    if level == 3:
        lines.append('Edit by exact text replacement (old must occur exactly once): '
                     '{"op": "edit", "path": "module.py", "old": "    assert x == 1", "new": "    assert y == 1"}')
    lines.append(f"Final action: {answer}")
    return "\n".join(lines)


REJECTS: Counter = Counter()  # why candidate tasks were regenerated; diagnostics for calibration and tests


def _reject(reason: str) -> None:
    REJECTS[reason] += 1
    return None


def make_flaky_task(task_id: str, level: int, seed: int, rng: random.Random, pool: tuple[str, ...]) -> Task:
    clean = rng.random() < CLEAN_RATE
    cause = rng.choice(CAUSES)
    for name in rng.sample(_eligible(pool), len(_eligible(pool))):
        item = corpus.ENTRIES[name]
        function = mutation.rename_locals(item.function_source, random.Random(rng.random()))
        taken = {n.name for n in ast.parse(function).body if isinstance(n, (ast.FunctionDef, ast.Assign))}
        rows = _rows(name, function, item.entry, rng)
        if len(rows) < 10 or len({json.dumps(o) for _, o in rows}) < 4:
            continue
        task = _build(task_id, level, seed, rng, name, item.entry, function, rows, cause, clean, taken)
        if task is not None:
            return task
    raise RuntimeError(f"no usable flaky task for {task_id} seed {seed}")


def _build(task_id, level, seed, rng, corpus_name, entry, function, rows, cause, clean, taken) -> Task | None:
    p = _Plan(entry=entry, rows=rows, rng=rng)
    mutant = None
    if level == 3 and not clean:
        found = _mutant(function, entry, rows, rng)
        if found is None:
            return _reject("no_mutant")
        mutant, p.zones = found
        p.zone = "kill"
    words = [w for w in TEST_WORDS if f"test_{w}" not in taken]
    names = {"state": rng.choice(STATE_NAMES), "state_key": rng.choice(KEY_NAMES),
             "store": rng.choice(STORE_NAMES), "helper": rng.choice(HELPER_NAMES)}
    if taken & set(names.values()):
        return _reject("name_clash")
    facility = FACILITY[cause]
    if facility == "store":
        pair = _store_pair(p)
        if pair is None:
            return _reject("no_store_pair")
        names["store_pair"], names["store_prefix"] = pair[:2], pair[2]
    n_tests = rng.choice(N_TESTS[level])
    others = [f for f in FACILITY.values() if f != facility]

    def extra() -> list[_Check]:
        if rng.random() >= P_MIX[level]:
            return []
        check = _snippet(p, rng.choice(others), "look" if rng.random() < P_LOOK[level] else "safe", names)
        return [check] if check else []

    culprit = _snippet(p, facility, "twin" if clean else "culprit", names)
    if culprit is None:
        return _reject("no_culprit_rows")
    p.zone = "spare"
    plan = [[culprit] + extra()]
    if facility in ("state", "store"):
        writer = _writer(p, cause, clean, names)
        if writer is None:
            return _reject("no_writer_rows")
        plan.append([writer] + extra())
    if rng.random() < P_SAME[level] and len(plan) < n_tests:
        same = _snippet(p, facility, "look" if rng.random() < P_LOOK[level] else "safe", names)
        if same:
            plan.append([same] + extra())
    while len(plan) < n_tests:
        check = _snippet(p, rng.choice(others), "look" if rng.random() < P_LOOK[level] else "safe", names)
        if check is None:
            return _reject("no_decoy_rows")
        plan.append([check] + extra())
    test_names = [f"test_{w}" for w in rng.sample(words, len(plan))]
    victim = test_names[0]
    order = rng.sample(range(len(plan)), len(plan))
    tests = [(test_names[i], plan[i]) for i in order]
    key_expr = _key_expr(p, cause, clean, names) if p.store_users else None
    region, boxes = _render(p, tests, names, key_expr)
    listed = [n for n, _ in tests]
    harness = _harness(listed, boxes)
    prefix = function.rstrip("\n") + "\n\n\n"
    source = prefix + region + harness
    if len(source.encode()) > 5000 or sandbox.static_gate(source):
        return _reject("size_or_gate")
    if any(word in region.replace(entry, "").lower() for word in TELLTALE):
        return _reject("telltale")
    graded = _combos(rng, len(listed), K_GRADE, shown=False)
    outcomes = _run(source, graded)
    if outcomes is None:
        return _reject("suite_unsettled")
    vary, broken = _varying(outcomes, listed)
    if broken or vary != ([] if clean else [victim]):
        return _reject("ground_truth")  # ground truth by execution disagrees with what was injected
    attributed = None
    if not clean:
        attribution = _attribute(prefix + region + _probe(harness, boxes, names), graded, victim)
        if attribution is None or attribution[1] != outcomes or attribution[0] != [cause]:
            return _reject("attribution")
        attributed = attribution[0][0]
    candidates = _combos(rng, len(listed), 12, shown=True)
    shown_out = _run(source, candidates)
    if shown_out is None:
        return _reject("shown_unsettled")
    shown = [c for c, o in zip(candidates, shown_out) if all(o.get(n) is True for n in listed)][:N_SHOWN]
    if len(shown) < N_SHOWN:
        return _reject("no_passing_shown_runs")
    private = {"corpus": corpus_name, "entry": entry, "original": source, "prefix": prefix, "harness": harness,
               "tests": listed, "victim": None if clean else victim, "cause": attributed, "graded": graded,
               "shown": shown, "mutants": [], "repair": None}
    if level == 3 and not clean:
        fixed_region, _ = _render(p, tests, names, key_expr, repaired=victim)
        fixed = prefix + fixed_region + harness
        old = _test_block(region, victim)
        new = _test_block(fixed_region, victim)
        if source.count(old) != 1 or not repair_ok(fixed, prefix, harness, mutant, graded, victim, listed):
            return _reject("repair")
        private["mutants"] = [{"source": mutant}]
        private["repair"] = {"old": old, "new": new}
    return Task(task_id=task_id, family="qe_flaky", difficulty=level, seed=seed,
                prompt=_prompt(entry, level, shown), files={MODULE: source}, private=private)


def _test_block(region: str, name: str) -> str:
    start = region.index(f"def {name}(env):\n")
    end = region.find("\n\n\n", start)
    return region[start:end]


def _structure(source: str, prefix: str, harness: str) -> str | None:
    if not (source.startswith(prefix) and source.endswith(harness) and len(source) >= len(prefix) + len(harness)):
        return None
    return source[len(prefix):len(source) - len(harness)]


def _all_pass(source: str, combos: list[list], listed: list[str]) -> bool:
    outcomes = _run(source, combos)
    return outcomes is not None and all(sorted(o) == sorted(listed) and all(v is True for v in o.values())
                                        for o in outcomes)


def repair_ok(source: str, prefix: str, harness: str, mutant: str, combos: list[list], victim: str,
              listed: list[str]) -> bool:
    """Every test passes on every graded run, and the repaired test fails on every run against the mutant."""
    region = _structure(source, prefix, harness)
    if region is None or not _all_pass(source, combos, listed):
        return False
    outcomes = _run(mutant.rstrip("\n") + "\n\n\n" + region + harness, combos)
    return outcomes is not None and all(o.get(victim) is False for o in outcomes)


def run_shown(task: Task, source: str) -> str:
    """Run the working copy's suite on the shown runs (the model's run_tests)."""
    p = task.private
    if _structure(source, p["prefix"], p["harness"]) is None:
        return "the code under test and the harness must stay unchanged; only the tests may be edited"
    results = _outputs(source, "run_suite", p["shown"])
    if results and all(r.kind == "rejected" for r in results):
        return "REJECTED: " + "; ".join(sandbox.static_gate(source))[:400]
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
    if p["victim"] is None:  # clean control: only "no flaky test" is right, and the suite must still pass
        if test is not None:
            return 0.0
        if task.difficulty == 3 and source != p["original"]:
            if _structure(source, p["prefix"], p["harness"]) is None or not _all_pass(source, p["graded"], p["tests"]):
                return 0.0
        return 1.0
    if test != p["victim"]:
        return 0.0
    if task.difficulty == 1:
        return 1.0
    if task.difficulty == 2:
        return 1.0 if cause == p["cause"] else 0.5
    mutant = p["mutants"][0]["source"]
    return 1.0 if repair_ok(source, p["prefix"], p["harness"], mutant, p["graded"], p["victim"], p["tests"]) else 0.5
