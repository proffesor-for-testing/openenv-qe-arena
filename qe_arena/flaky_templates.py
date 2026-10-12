"""qe_flaky check templates. Each template has a faulty and a safe parametrization with the same text shape,
and decoys imitate the flaky test's own template, so only reading what the code does tells them apart."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .flaky import FRAC_SWEEP, TICK

P_TWIN = 0.85  # a decoy on the flaky test's facility uses the very template the flaky test uses
SINGLE = ("draw", "frac", "now", "pool")
KEY_NAMES = ("input", "case", "args", "sample", "probe", "subject", "item", "seed_case", "target", "query")
CONSTS = ("0.1", "0.2", "0.3", "0.5", "0.25", "0.125", "0.7", "1.1", "2", "3", "4", "5", "8", "10", "16", "0.05",
          "1.5", "0.75", "6", "7")
FLOAT_FORMS = (  # (setup line or "", asserted expression); {a} {b} are constants, {p} a parts list, {q} its sum
    ("", "(w + {a}) - {a} == w"), ("", "w * {a} / {a} == w"), ("", "w / {a} * {a} == w"),
    ("", "(w + {a}) + {b} == w + ({a} + {b})"), ("", "w * {a} * {b} == w * ({a} * {b})"),
    ("total = w * {a}", "total / {a} == w"), ("total = w / {a}", "total * {a} == w"),
    ("total = w + {a}", "total - {a} == w"), ("parts = [{p}]", "sum(parts) == {q}"),
    ("", "w * {a} == {a} * w"), ("", "w + {a} == {a} + w"), ("total = w + {a}", "total == {a} + w"),
)
POW2 = frozenset({"0.5", "0.25", "0.125", "2", "4", "8", "16"})
_r = random.Random(20261012)
FRAC_SAMPLE = [_r.randrange(1, 2147483647) / 2147483647 for _ in range(3000)]
FRAC_GRID = FRAC_SWEEP  # values env.frac can actually produce
_FLOAT_CLASS: dict[tuple[str, str], str] = {}



@dataclass
class Check:
    lines: list[str]
    args: list[list]                  # every input the check calls the function with
    repair: list[str] | None = None   # the canonical repair of a faulty check


@dataclass
class Plan:
    entry: str
    rows: list[tuple[list, object]]
    rng: random.Random
    names: dict
    low: bool = False
    used: set = field(default_factory=set)
    zones: dict | None = None   # d3: {"kill": inputs the mutant returns another value for, "spare": same value}
    zone: str = ""
    state: dict = field(default_factory=dict)        # dict key -> initial args
    store: list = field(default_factory=list)        # (args, output) of every helper call
    key: tuple = ("full",)
    hint: tuple = ("", "")           # (facility, template) of the flaky test, which decoys imitate
    form: int | None = None          # its float form, when it is a float check

    def call(self, args: list) -> str:
        return f"{self.entry}({', '.join(repr(a) for a in args)})"

    def free(self, ok=None) -> list:
        out = [r for r in self.rows if json.dumps(r[0]) not in self.used and (ok is None or ok(r))
               and (self.zones is None or not self.zone or json.dumps(r[0]) in self.zones[self.zone])]
        self.rng.shuffle(out)
        return out

    def distinct(self, count: int, ok=None) -> list | None:
        chosen = []
        for row in self.free(ok):
            if all(not (row[1] == c[1]) for c in chosen):
                chosen.append(row)
                if len(chosen) == count:
                    return self.mark(chosen)
        return None

    def same(self, count: int, ok=None) -> list | None:
        groups: dict[str, list] = {}
        for row in self.free(ok):
            group = groups.setdefault(repr(row[1]), [])
            if all(row[1] == g[1] for g in group):
                group.append(row)
                if len(group) == count:
                    return self.mark(group)
        return None

    def mark(self, rows):
        self.used.update(json.dumps(r[0]) for r in rows)
        return rows

    def keyed(self, args: list):
        kind, text = self.key[0], repr(tuple(args))
        if kind == "prefix":
            return text[:self.key[1]]
        return repr(args[0]) if kind == "first" else len(text) if kind == "length" else text

    def store_ok(self, row) -> bool:
        return all(not (self.keyed(a) == self.keyed(row[0])) or o == row[1] for a, o in self.store)


def _lit(values: list) -> str:
    return ", ".join(repr(v) for v in values)


def _float_class(setup: str, expr: str) -> str:
    """safe (always holds), normal or low (fails on 8-88% or 2-10% of values, and on the sweep grid), or other."""
    if (setup, expr) not in _FLOAT_CLASS:
        name, _, value = setup.partition(" = ")
        body = f"(lambda {name}: {expr})({value})" if setup else expr
        fn = eval(compile(f"lambda w: {body}", "<float>", "eval"), {"__builtins__": {}, "sum": sum})  # noqa: S307
        rate = sum(map(fn, FRAC_SAMPLE)) / len(FRAC_SAMPLE)
        grid_fails = not all(map(fn, FRAC_GRID))
        _FLOAT_CLASS[(setup, expr)] = ("safe" if rate == 1 and not grid_fails else
                                       "normal" if 0.12 <= rate <= 0.92 and grid_fails else
                                       "low" if 0.9 <= rate <= 0.98 and grid_fails else "other")
    return _FLOAT_CLASS[(setup, expr)]


def _float_expr(rng: random.Random, want: str, form: int | None = None) -> tuple[str, str, int] | None:
    pow2 = rng.random() < 0.5  # exact and inexact checks both come with power-of-two and other constants
    pool = [c for c in CONSTS if (c in POW2) == pow2]
    for attempt in range(300):
        index = form if form is not None and attempt < 100 else rng.randrange(len(FLOAT_FORMS))
        setup, expr = FLOAT_FORMS[index]
        a, b = rng.sample(pool if attempt < 200 else CONSTS, 2)
        terms = ["w", a, b]
        rng.shuffle(terms)
        fields = {"a": a, "b": b, "p": ", ".join(terms), "q": " + ".join(rng.sample(["w", a, b], 3))}
        setup, expr = setup.format(**fields), expr.format(**fields)
        if _float_class(setup, expr) == want:
            return setup, expr, index
    return None


def single_check(p: Plan, facility: str, mode: str) -> Check | None:
    """A check on draw, frac, now or pool. mode: bad, safe (same template, safe values) or filler."""
    E, rng, bad = p.entry, p.rng, mode == "bad"
    tmpl = "filler" if mode == "filler" else "ternary" if bad and p.low and facility == "draw" else \
        p.hint[1] if p.hint[0] == facility and rng.random() < P_TWIN else rng.choice(
            {"draw": ("list", "ternary", "table"), "frac": ("expr",), "now": ("stamp", "window", "deadline"),
             "pool": ("list", "first")}[facility])
    imitate = not bad and (facility, tmpl) == p.hint  # a decoy shaped exactly like the flaky check
    if not p.hint[0]:
        p.hint = (facility, tmpl)
    if facility == "draw":
        if tmpl == "filler":
            rows = p.distinct(1)
            return rows and Check(["count = 1 + env.draw(3)", f"assert [{p.call(rows[0][0])}] * count == "
                                   f"[{rows[0][1]!r}] * count"], [rows[0][0]])
        m = rng.choice((2, 3)) if tmpl == "list" else 2
        rows = p.same(m) if not bad and (imitate or rng.random() < 0.5) else None
        same = rows is not None
        rows = rows or p.distinct(m)
        if rows is None:
            return None
        (a, oa), (b, ob) = rows[0], rows[1]
        want = f"== {oa!r}" if bad or same else f"in ({_lit([r[1] for r in rows])})"
        if bad and rng.random() < 0.4:  # a faulty check spelled like a safe one: the tuple misses one branch
            other = next((o for _, o in p.rows if all(not (o == r[1]) for r in rows)), None)
            if other is not None:
                want = f"in ({_lit([oa, other] if rng.random() < 0.5 else [other, oa])})"
        repair = [f"assert {p.call(a)} == {oa!r}"] if bad else None
        if tmpl == "list":
            items = [r[0] for r in rows]
            rng.shuffle(items)
            return Check([f"args = [{_lit(items)}][env.draw({m})]", f"assert {E}(*args) {want}"], items, repair)
        if tmpl == "ternary":
            k = rng.choice((12, 16, 20, 25, 30, 40)) if bad and p.low else rng.choice((2, 3, 4, 12, 20) if not bad else
                                                                                       (2, 3, 4))
            return Check([f"args = {a!r} if env.draw({k}) else {b!r}", f"assert {E}(*args) {want}"], [a, b], repair)
        pairs = [(a, oa), (b, oa if bad else ob)]
        rng.shuffle(pairs)
        table = ", ".join(f"({x!r}, {y!r})" for x, y in pairs)
        return Check([f"args, want = [{table}][env.draw(2)]", f"assert {E}(*args) == want"], [a, b], repair)
    if facility == "pool":
        if tmpl == "filler":
            rows = p.distinct(2)
            return rows and Check([f"got = [{E}(*a) for a in env.pool([{_lit([r[0] for r in rows])}])]",
                                   f"assert len(got) == 2 and {rows[0][1]!r} in got and {rows[1][1]!r} in got"],
                                  [r[0] for r in rows])
        rows = p.same(2) if not bad and ((imitate and tmpl == "first") or rng.random() < 0.5) else None
        same = rows is not None
        rows = rows or p.distinct(2)
        if rows is None:
            return None
        (a, oa), (b, ob) = rows
        if tmpl == "list":
            listed = f"[{E}(*a) for a in env.pool([{a!r}, {b!r}])]"
            if bad or same:
                return Check([f"got = {listed}", f"assert got == [{oa!r}, {ob!r}]"], [a, b],
                             [f"got = [{E}(*a) for a in [{a!r}, {b!r}]]", f"assert got == [{oa!r}, {ob!r}]"]
                             if bad else None)
            ordered = sorted([oa, ob], key=repr)
            return Check([f"got = sorted({listed}, key=repr)", f"assert got == [{_lit(ordered)}]"], [a, b])
        index = rng.choice((0, -1))
        target = (a, oa) if index == 0 else (b, ob)
        want = f"== {target[1]!r}" if bad or same else f"in ({oa!r}, {ob!r})"
        other = next((o for _, o in p.rows if not (o == oa) and not (o == ob)), None)
        if bad and other is not None and rng.random() < 0.4:
            want = f"in ({_lit([target[1], other] if rng.random() < 0.5 else [other, target[1]])})"
        return Check([f"first = env.pool([{a!r}, {b!r}])[{index}]", f"assert {E}(*first) {want}"], [a, b],
                     [f"assert {p.call(target[0])} == {target[1]!r}"] if bad else None)
    rows = p.distinct(1)
    if rows is None:
        return None
    (a, oa), = rows
    if facility == "frac":
        if tmpl == "filler":
            expr = rng.choice(("0 < w < 1", "abs((w + 0.1) - 0.1 - w) < 1e-9"))
            return Check(["w = env.frac()", f"assert {p.call(a)} == {oa!r} and {expr}"], [a])
        found = _float_expr(rng, ("low" if p.low else "normal") if bad else "safe", p.form if imitate else None)
        if found is None:
            return None
        setup, expr, form = found
        if p.form is None:
            p.form = form
        lines = ["w = env.frac()"] + ([setup] if setup else []) + [f"assert {p.call(a)} == {oa!r} and {expr}"]
        return Check(lines, [a], [f"assert {p.call(a)} == {oa!r}"] if bad else None)
    bucket = rng.choice((4000, 5000, 10000) if bad and p.low else (500, 1000, 2000, 2500) if bad else
                        (500, 1000, 2000, 2500, 4000, 5000, 10000))
    got = [f"got = {p.call(a)}"]
    repair = got + [f"assert got == {oa!r}"] if bad else None
    if tmpl == "filler":
        return Check(["start = env.now()"] + got + [f"assert got == {oa!r} and env.now() - start < {bucket}"], [a])
    B, T = bucket, TICK
    # (first line, check): the second read is always the first plus the tick, so each pair is faulty or safe
    faulty = {"stamp": [(f"stamp = env.now() // {B}", f"env.now() // {B} == stamp"),
                        (f"stamp = (env.now() + {T}) // {B}", f"(env.now() - {T}) // {B} == stamp"),
                        (f"stamp = (env.now() - {T}) // {B}", f"env.now() // {B} == stamp")],
              "window": [("start = env.now()", f"env.now() // {B} == start // {B}"),
                         (f"start = env.now() - {T}", f"env.now() // {B} == start // {B}")],
              "deadline": [(f"deadline = (env.now() // {B} + 1) * {B}", "env.now() < deadline"),
                           (f"deadline = (env.now() - {T}) // {B} * {B} + {B}", "env.now() < deadline")]}
    safe = {"stamp": [(f"stamp = (env.now() + {T}) // {B}", f"env.now() // {B} == stamp"),
                      (f"stamp = env.now() // {B}", f"(env.now() - {T}) // {B} == stamp"),
                      (f"stamp = env.now() // {B}", f"env.now() // {B} - stamp <= 1"),
                      (f"stamp = (env.now() - {T}) // {B}", f"env.now() // {B} >= stamp")],
            "window": [("start = env.now()", f"(env.now() - {T}) // {B} == start // {B}"),
                       ("start = env.now()", f"env.now() // {B} - start // {B} <= 1")],
            "deadline": [(f"deadline = (env.now() + {T}) // {B} * {B} + {B}", "env.now() < deadline"),
                         (f"deadline = env.now() + {B}", "env.now() < deadline")]}
    options = faulty[tmpl][:1 if p.low and tmpl != "stamp" else None] if bad else safe[tmpl]
    if bad and B == 500:  # a two-tick window on a 500 ms bucket would fail on every run
        options = [o for o in options if f"- {T}" not in o[0]]
    if bad and p.low and tmpl == "stamp":
        options = options[:2]  # keep a low-rate flake's window at one tick
    first, check = rng.choice(options)
    return Check([first] + got + [f"assert got == {oa!r} and {check}"], [a], repair)


def state_unit(p: Plan, mode: str, key: str, single: bool = False) -> list[list[Check]] | None:
    """Tests on the module-level dict: [reader, writer] or a single test. mode: bad, twin, safe."""
    E, rng, D = p.entry, p.rng, p.names["state"]
    if mode == "safe" and (single or rng.random() < 0.4):
        rows = p.distinct(1)
        if rows is None:
            return None
        (c, oc), = rows
        p.state[key] = c
        if rng.random() < 0.5:
            return [[Check([f"{D}[{key!r}] = {c!r}", f"assert {E}(*{D}[{key!r}]) == {oc!r}"], [c])]]
        if rng.random() < 0.5:
            return [[Check([f"case = {D}[{key!r}]", f"assert {E}(*case) == {oc!r}"], [c])]]
        return [[Check([f"assert {E}(*{D}[{key!r}]) == {oc!r}"], [c])]]
    reader_zone = p.zone
    rows = p.distinct(1)
    if rows is None:
        return None
    (a, oa), = rows
    p.zone = "spare" if p.zones else ""
    style = "set" if mode == "bad" or (p.hint[0] == "state" and rng.random() < P_TWIN) else \
        rng.choice(("set", "restore", "finally"))
    rows = p.same(1, ok=lambda r: r[1] == oa) if style == "set" and mode != "bad" else \
        p.distinct(1, ok=lambda r: not (r[1] == oa))
    p.zone = reader_zone
    if rows is None:
        return None
    (b, ob), = rows
    p.state[key] = a
    reader = [f"assert {E}(*{D}[{key!r}]) == {oa!r}"] if rng.random() < 0.5 else \
        [f"case = {D}[{key!r}]", f"assert {E}(*case) == {oa!r}"]
    write = rng.choice((f"{D}[{key!r}] = {b!r}", f"{D}.update({{{key!r}: {b!r}}})"))
    check = f"assert {E}(*{D}[{key!r}]) == {ob!r}"
    if style == "set":
        writer = [write, check]
    elif style == "restore":
        writer = [f"saved = {D}[{key!r}]", write, check, f"{D}[{key!r}] = saved"]
    else:
        writer = [f"saved = {D}[{key!r}]", write, "try:", f"    {check}", "finally:", f"    {D}[{key!r}] = saved"]
    repair = [f"assert {p.call(a)} == {oa!r}"] if mode == "bad" else None
    return [[Check(reader, [a], repair)], [Check(writer, [b])]]


def store_unit(p: Plan, mode: str, single: bool = False) -> list[list[Check]] | None:
    """Tests calling the memoizing helper: [reader, warm-up caller] or a lone reader. mode: bad, twin, safe."""
    rng, H = p.rng, p.names["helper"]
    if mode == "bad":
        a, b = p.names["pair"]
    else:
        rows = p.distinct(1, ok=p.store_ok)
        if rows is None:
            return None
        a = rows[0]
        p.store.append(a)
        b = None
        if mode == "twin" or (not single and rng.random() < 0.6):
            reader_zone, p.zone = p.zone, "spare" if p.zones else ""
            hit = p.same(1, ok=lambda r: r[1] == a[1] and r[1] is not None and p.keyed(r[0]) == p.keyed(a[0]))
            rows = [hit[0]] if hit else p.distinct(1, ok=lambda r: r[1] is not None and p.store_ok(r))
            p.zone = reader_zone
            if rows is None:
                return None
            b = rows[0]
            p.store.append(b)
    call = f"{H}({_lit(a[0])})"
    reader = [f"assert {call} == {a[1]!r}"] if rng.random() < 0.5 else [f"got = {call}", f"assert got == {a[1]!r}"]
    tests = [[Check(reader, [a[0]], [f"assert {p.call(a[0])} == {a[1]!r}"] if mode == "bad" else None)]]
    if b is not None:
        warm = f"{H}({_lit(b[0])})"
        tests.append([Check([rng.choice((f"assert {warm} is not None", f"assert len(repr({warm})) > 0", warm))],
                            [b[0]])])
    return tests


def choose_store_key(p: Plan, bad: bool) -> bool:
    """Choose the helper's key; for a cache-leak culprit, a key under which two inputs collide."""
    kinds = ["prefix", "first", "length"]
    p.rng.shuffle(kinds)
    if not bad:
        kind = p.rng.choice(kinds + ["full"])
        p.key = ("prefix", p.rng.randrange(5, 15)) if kind == "prefix" else (kind,)
        return True
    kill = p.free(lambda r: r[1] is not None) if not p.zones else \
        [r for r in p.free(lambda r: r[1] is not None) if json.dumps(r[0]) in p.zones["kill"]]
    spare = [r for r in p.rows if json.dumps(r[0]) not in p.used and r[1] is not None
             and (not p.zones or json.dumps(r[0]) in p.zones["spare"])]
    for kind in kinds:
        for a in kill:
            for b in spare:
                if a is b or a[1] == b[1]:
                    continue
                ka, kb = repr(tuple(a[0])), repr(tuple(b[0]))
                if kind == "prefix":
                    common = next((i for i in range(min(len(ka), len(kb))) if ka[i] != kb[i]), None)
                    if common is None or common < 4:
                        continue
                    p.key = ("prefix", common)
                elif kind == "first" and not (len(a[0]) > 1 and repr(a[0][0]) == repr(b[0][0])):
                    continue
                elif kind == "length" and len(ka) != len(kb):
                    continue
                else:
                    p.key = (kind,)
                p.mark([a, b])
                p.names["pair"] = (a, b)
                p.store += [a, b]
                return True
    return False


def key_text(key: tuple) -> str:
    return {"full": "repr(args)", "prefix": f"repr(args)[:{key[-1]}]", "first": "repr(args[0])",
            "length": "len(repr(args))"}[key[0]]


def decoy_unit(p: Plan, facility: str, mode: str, room: int) -> list[list[Check]] | None:
    if facility == "state":
        key = p.rng.choice([k for k in KEY_NAMES if k not in p.state and k != p.names.get("state_key")])
        return state_unit(p, "safe", key, single=room < 2)
    if facility == "store":
        return store_unit(p, "safe", single=room < 2)
    check = single_check(p, facility, mode)
    return [[check]] if check else None
