"""qe_flaky generation. Every test is built from templates that each have a faulty and a safe parametrization
(the same text shape with different values), so the flaky test can only be told apart by reading what the
code does. Ground truth, cause attribution and repairability are then established by execution in ``flaky``."""

from __future__ import annotations

import json
import random
from collections import Counter

from . import corpus, mutation, sandbox
from .flaky import (CAUSES, FACILITY, SEED_SPLIT, MAX_GRADED_BYTES, attribute, combos, gate_region,
                    graded_source, prompt, repair_ok, run, shown_harness, sweep_calls, top_names, victim_block)
from .flaky_templates import (KEY_NAMES, SINGLE, Plan, choose_store_key, decoy_unit, key_text,
                              single_check, state_unit, store_unit)
from .tasks import MODULE, Task, _outputs

CLEAN_RATE = 0.15
LOW_RATE = 0.25     # share of draw/now/frac flakes that fail only 2-10% of runs
K_GRADE, K_LOW = 24, 96
N_SHOWN = 2
N_TESTS = {1: (4,), 2: (4, 5), 3: (4, 5)}
P_MIX = {1: 0.3, 2: 0.4, 3: 0.3}   # a test carries a second check on another facility
P_SAME = 0.95                       # a decoy reuses the flaky test's facility ...
N_SAME = {1: 2, 2: 2, 3: 2}          # at most this many such decoys
P_FILLER = 0.15                      # a decoy uses a safe-only form
MAX_FUNCTION_BYTES = 1800
MAX_LITERAL = 48
TELLTALE = ("flak", "random", "cache", "memo", "leak", "shared", "unorder", "float", "clock", "boundary",
            "order", "victim", "pollut", "culprit", "race", "time")
TEST_WORDS = ("amber", "basalt", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper",
              "kestrel", "lagoon", "marble", "nectar", "onyx", "prairie", "quartz", "raven", "sierra", "tundra",
              "umber", "violet", "willow", "zephyr", "alder", "birch", "comet", "dune", "fern", "glacier")
STATE_NAMES = ("DEFAULTS", "SETTINGS", "FIXTURE", "PARAMS", "PROFILE", "BASELINE")
STORE_NAMES = ("STORE", "TABLE", "LEDGER", "SLOTS", "INDEX", "BANK")
HELPER_NAMES = ("lookup", "fetch", "resolve", "obtain", "evaluate", "answer_for")

REJECTS: Counter = Counter()  # why candidate tasks were regenerated; diagnostics for calibration and tests


def _reject(reason: str) -> None:
    REJECTS[reason] += 1
    return None


def _rows(name: str, source: str, entry: str, rng: random.Random) -> list[tuple[list, object]]:
    inputs = corpus.gen_inputs(name, rng.randrange(2**31), 64)
    results = _outputs(source, entry, inputs)
    rows, seen, size = [], set(), 0
    for args, result in zip(inputs, results):
        text, out = json.dumps(args), json.dumps(result.value)
        if result.kind != "ok" or not args or text in seen or len(text) > MAX_LITERAL or len(out) > MAX_LITERAL:
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
    # The tests compare Python values; keep only rows whose literal equals what the function returns there.
    check = source + f"\n\nROWS = {rows!r}\n\n\ndef same(i):\n    a, o = ROWS[i]\n    return {entry}(*a) == o\n"
    verdicts = _outputs(check, "same", [[i] for i in range(len(rows))])
    return [row for row, v in zip(rows, verdicts) if v.kind == "ok" and v.value is True]


def _mutant(function: str, entry: str, rows: list, rng: random.Random) -> tuple[str, dict] | None:
    """A d3 reference mutant and its zones: the faulty check draws only inputs the mutant returns another value
    for (never a raise, which would fail a test that asserts nothing), every other test only unchanged ones."""
    calls = [a for a, _ in rows]
    reference = _outputs(function, entry, calls)
    sites = mutation.enumerate_sites(function, mutation.CLASSIC_OPERATORS)  # realistic operators would shift tasks
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
        if len(zones["kill"]) >= 3 and len(zones["spare"]) >= 12:
            return mutant, zones
    return None


def build_task(task_id: str, level: int, seed: int, rng: random.Random, pool: tuple[str, ...]) -> Task:
    clean = rng.random() < CLEAN_RATE
    cause = rng.choice(CAUSES)
    low = FACILITY[cause] in ("draw", "now", "frac") and not clean and rng.random() < LOW_RATE
    eligible = [n for n in sorted(pool) if len(corpus.ENTRIES[n].function_source) <= MAX_FUNCTION_BYTES
                and "flak" not in corpus.ENTRIES[n].function_source.lower()]
    for name in rng.sample(eligible, len(eligible)):
        item = corpus.ENTRIES[name]
        function = mutation.rename_locals(item.function_source, random.Random(rng.random()))
        rows = _rows(name, function, item.entry, rng)
        if len(rows) < 12 or len({repr(o) for _, o in rows}) < 4:
            continue
        task = _build(task_id, level, seed, rng, name, item.entry, function, rows, cause, clean, low)
        if task is not None:
            return task
    raise RuntimeError(f"no usable flaky task for {task_id} seed {seed}")


def _build(task_id, level, seed, rng, corpus_name, entry, function, rows, cause, clean, low) -> Task | None:
    protected = top_names(function)
    names = {"state": rng.choice(STATE_NAMES), "state_key": rng.choice(KEY_NAMES),
             "store": rng.choice(STORE_NAMES), "helper": rng.choice(HELPER_NAMES)}
    words = [w for w in TEST_WORDS if f"test_{w}" not in protected]
    if protected & set(names.values()):
        return _reject("name_clash")
    p = Plan(entry=entry, rows=rows, rng=rng, names=names, low=low)
    mutant = None
    if level == 3 and not clean:
        found = _mutant(function, entry, rows, rng)
        if found is None:
            return _reject("no_mutant")
        mutant, p.zones = found
        p.zone = "kill"
    facility = FACILITY[cause]
    if not choose_store_key(p, facility == "store" and not clean):
        return _reject("no_store_pair")
    mode = "twin" if clean else "bad"
    if facility in ("state", "store"):
        p.hint = (facility, "pair")
    if facility == "state":
        victim_unit = state_unit(p, mode, names["state_key"])
    elif facility == "store":
        victim_unit = store_unit(p, mode)
    else:
        check = single_check(p, facility, mode if not clean else "safe")
        victim_unit = [[check]] if check else None
    if victim_unit is None:
        return _reject("no_culprit_rows")
    if mutant is not None:  # an order-insensitive repair must not let a mutant that permutes outputs through
        culprit = victim_unit[0][0]
        changed = _outputs(mutant, entry, culprit.args)
        reference = [o for a in culprit.args for r, o in rows if r == a][:len(culprit.args)]
        if sorted(map(repr, (c.value for c in changed))) == sorted(map(repr, reference)):
            return _reject("mutant_permutes_culprit_outputs")
    p.zone = "spare" if p.zones else ""
    plan = list(victim_unit)
    n_tests = rng.choice(N_TESTS[level])
    want_same = N_SAME[level] if rng.random() < P_SAME else 0
    for _ in range(12):
        if len(plan) >= n_tests:
            break
        target = facility if want_same else rng.choice(SINGLE + ("state", "store"))
        want_same = max(0, want_same - 1)
        unit = decoy_unit(p, target, "filler" if rng.random() < P_FILLER else "safe", n_tests - len(plan))
        if unit and len(plan) + len(unit) <= n_tests:
            plan.extend(unit)
    if len(plan) != n_tests:
        return _reject("no_decoy_rows")
    for checks in plan:
        if rng.random() < P_MIX[level]:
            kind = rng.choice([f for f in SINGLE if f != facility] + ["state"])
            if kind == "state":  # a test that writes and reads its own dict entry, harmless wherever it sits
                key = [k for k in KEY_NAMES if k not in p.state and k != names["state_key"]]
                unit = state_unit(p, "safe", rng.choice(key), single=True) if key else None
                extra = unit[0][0] if unit else None
            else:
                extra = single_check(p, kind, "safe")
            if extra:
                checks.append(extra)
    test_names = [f"test_{w}" for w in rng.sample(words, len(plan))]
    victim = test_names[0]
    order = rng.sample(range(len(plan)), len(plan))
    tests = [(test_names[i], plan[i]) for i in order]
    region, boxes = _render(p, tests, names)
    listed = [n for n, _ in tests]
    harness = shown_harness(listed, boxes)
    prefix = function.rstrip("\n") + "\n\n\n"
    source = prefix + region + harness
    graded_src = graded_source(prefix, region, listed, boxes)
    if graded_src is None or max(len(source.encode()), len(graded_src.encode())) > MAX_GRADED_BYTES \
            or sandbox.static_gate(graded_src):
        return _reject("size_or_gate")
    if gate_region(region, protected) or any(w in region.replace(entry, "").lower() for w in TELLTALE):
        return _reject("region_gate_or_telltale")
    graded = combos(rng, len(listed), K_LOW if low else K_GRADE, shown=False)
    outcomes = run(graded_src, graded)
    if outcomes is None or run(source, graded) != outcomes:
        return _reject("suite_unsettled_or_shown_harness_differs")
    vary = [n for n in listed if {o.get(n) for o in outcomes} == {True, False}]
    broken = [n for n in listed if n not in vary and {o.get(n) for o in outcomes} != {True}]
    if broken or vary != ([] if clean else [victim]):
        return _reject("ground_truth")  # ground truth by execution disagrees with what was injected
    idx = {"state": [i for i, b in enumerate(boxes) if b == names["state"]],
           "store": [i for i, b in enumerate(boxes) if b == names["store"]]}
    if not clean:
        attribution = attribute(graded_src, graded, victim, idx)
        if attribution is None or attribution[1] != outcomes or attribution[0] != [cause]:
            return _reject("attribution")
    sweep = sweep_calls(random.Random(rng.random()))  # the task's own secret stream
    if not clean and facility in SINGLE:  # the sweep must catch every faulty template on its own
        swept = run(graded_src, [[victim] + s for s in sweep], entry="sweep")
        if swept is None or all(o.get(victim) for o in swept):
            return _reject("sweep_blind")
    candidates = combos(rng, len(listed), 12, shown=True)
    shown_out = run(source, candidates)
    if shown_out is None:
        return _reject("shown_unsettled")
    shown = [c for c, o in zip(candidates, shown_out) if all(o.get(n) is True for n in listed)][:N_SHOWN]
    if len(shown) < N_SHOWN:
        return _reject("no_passing_shown_runs")
    task = Task(task_id=task_id, family="qe_flaky", difficulty=level, seed=seed,
                prompt=prompt(entry, level, shown), files={MODULE: source},
                private={"corpus": corpus_name, "entry": entry, "original": source, "prefix": prefix,
                         "region": region, "harness": harness, "tests": listed, "boxes": boxes,
                         "protected": sorted(protected), "victim": None if clean else victim,
                         "cause": None if clean else cause, "band": None if clean else "low" if low else "normal",
                         "graded": graded, "shown": shown, "sweep": sweep, "mutants": [], "repair": None})
    if level == 3 and not clean:
        fixed_region, _ = _render(p, tests, names, repaired=victim)
        old, new = victim_block(region, victim), victim_block(fixed_region, victim)
        task.private["mutants"] = [{"source": mutant}]
        if source.count(old) != 1 or not repair_ok(task, region.replace(old, new, 1), isolated=False):
            return _reject("repair")
        task.private["repair"] = {"old": old, "new": new}
    return task


def _render(p: Plan, tests: list, names: dict, repaired: str | None = None) -> tuple[str, list[str]]:
    """The tests region (dict, helper, tests) and the names of its module-level dicts."""
    parts, boxes = [], []
    if p.state:
        parts.append(f"{names['state']} = {dict(sorted(p.state.items()))!r}")
        boxes.append(names["state"])
    if p.store:
        S, H = names["store"], names["helper"]
        parts.append(f"{S} = {{}}\n\n\ndef {H}(*args):\n    key = {key_text(p.key)}\n    if key not in {S}:\n"
                     f"        {S}[key] = {p.entry}(*args)\n    return {S}[key]")
        boxes.append(S)
    for name, checks in tests:
        body = []
        for check in checks:
            body.extend("    " + line for line in (check.repair if repaired == name and check.repair else check.lines))
        parts.append(f"def {name}(env):\n" + "\n".join(body))
    return "\n\n\n".join(parts) + "\n\n\n", boxes


__all__ = ["build_task", "REJECTS", "SEED_SPLIT"]
