"""Task generation and grading for the quality-engineering families.

A task is fully determined by ``(task_id, seed)``. ``private`` holds verifier
material (reference outputs, mutants, hidden inputs) and MUST NOT be sent in an
observation. Every reference output comes from running our own corpus code in
the sandbox, so the oracle and the candidate share one execution semantics.
"""

from __future__ import annotations

import itertools
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from . import corpus, mutation, sandbox

FAMILIES = ("qe_bugfix", "qe_mutant_kill")
DOMAIN = "software engineering"
TASK_IDS = [f"{family}-d{level}" for family in FAMILIES for level in (1, 2, 3)]

HIDDEN_INPUTS = 48
SUBTLE_RATE = 0.15  # a fault or mutant that at most this share of hidden inputs reveals
FAULTS = {1: 1, 2: 2, 3: 3}  # qe_bugfix faults per level; the prompt never states the count
MIN_SUBTLE = {1: 0, 2: 1, 3: 2}
MIN_REVEALING_D1 = 3  # distinct hidden calls that reveal a d1 fault; the bug report names one of them
MAX_TESTS = {1: 1, 2: 1, 3: 3}
MAX_ARGS_BYTES = 2048
MODULE = "module.py"


@dataclass
class Task:
    task_id: str
    family: str
    difficulty: int
    seed: int
    prompt: str
    files: dict[str, str]
    private: dict[str, Any] = field(repr=False)


def format_task_id(family: str, difficulty: int) -> str:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family!r}")
    if type(difficulty) is not int or difficulty not in (1, 2, 3):
        raise ValueError("difficulty must be 1, 2 or 3")
    return f"{family}-d{difficulty}"


def parse_task_id(task_id: str) -> tuple[str, int]:
    if type(task_id) is not str or len(task_id) > 64:
        raise ValueError("task_id must be a short string")
    match = re.fullmatch(r"(qe_[a-z_]+)-d([123])", task_id)
    if match is None or match.group(1) not in FAMILIES:
        raise ValueError(f"invalid task_id: {task_id!r}")
    return match.group(1), int(match.group(2))


def _show(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(", ", ": "))


def _call_text(entry: str, args: list) -> str:
    return f"{entry}({', '.join(_show(a) for a in args)})"


MAX_SHOWN = 200  # characters of any one value shown in an observation


def _clip(text: str) -> str:
    return text if len(text) <= MAX_SHOWN else text[:MAX_SHOWN] + f"... ({len(text)} chars)"


def _result_text(result: sandbox.CallResult) -> str:
    if result.kind == "ok":
        return _clip(_show(result.value))
    if result.kind == "exc":
        return f"raises {_clip(str(result.value))}"
    return result.kind


SETTLED = ("ok", "exc")
# Limits of our sandbox, not behaviour of the code: a 256 MiB cap or the stack depth can make one side
# raise these where the other merely ran longer. They are never a behavioural difference.
RESOURCE_EXCEPTIONS = ("MemoryError", "RecursionError")


def _settled(result: sandbox.CallResult) -> bool:
    return result.kind == "ok" or (result.kind == "exc" and result.value not in RESOURCE_EXCEPTIONS)


def _outputs(source: str, entry: str, calls: list[list]) -> list[sandbox.CallResult]:
    """Run calls once: fuel metering makes every outcome deterministic, so machine load cannot decide one.

    Callers treat any unsettled result (fuel, resource, crash, skipped) as unusable, never as a
    behavioural difference. ``infra`` (the sandbox worker itself failed; ``timeout`` is its legacy
    alias) is the only transient kind: the batch is run once more, and if it still fails this raises
    InfraError instead of returning anything a grade could turn into a reward.
    """
    results = sandbox.run_calls(source, entry, calls)
    if any(r.kind in ("infra", "timeout") for r in results):
        results = sandbox.run_calls(source, entry, calls)
        failed = [r for r in results if r.kind in ("infra", "timeout")]
        if failed:
            raise sandbox.InfraError(f"sandbox infrastructure failed twice: {failed[0].error}"[:300])
    return results


def _killing(reference: list, outputs: list) -> list[int]:
    """Inputs on which both sides settled and their results differ."""
    return [i for i, (r, o) in enumerate(zip(reference, outputs))
            if _settled(r) and _settled(o) and not sandbox.compare(r, o)]


def _usable(results: list) -> bool:
    return all(_settled(r) for r in results)


def _subsets_all_fail(original: str, sites: list, entry: str, inputs: list, reference: list) -> bool:
    """Every non-empty proper subset of the faults must still fail some hidden input.

    Otherwise fixing the complement passes the hidden tests: faults that cancel
    each other on the hidden inputs would pay a partial fix 1.0.
    """
    for size in range(1, len(sites)):
        for subset in itertools.combinations(sites, size):
            outputs = _outputs(mutation.apply(original, list(subset)), entry, inputs)
            if not _usable(outputs) or not _killing(reference, outputs):
                return False
    return True


REPLY_FORMAT = (
    "Reply with exactly one JSON action object and nothing else: no prose, no markdown, "
    "and do not repeat the schema."
)


def _bugfix_prompt(entry: str, visible: list[str], report: str) -> str:
    lines = [
        f"Faults were introduced into `{entry}` in {MODULE}. The docstring is the specification "
        "and is correct; the code is not. There may be more than one fault, and the number is not given.",
        report,
        "Fix the code so it matches the specification for every input. Hidden tests decide the reward, "
        "and a partial fix earns nothing.",
        "Visible tests:",
        *visible,
        "",
        REPLY_FORMAT,
        'First action: {"op": "read", "path": "*"}',
        'Edit by exact text replacement (old must occur exactly once): '
        '{"op": "edit", "path": "module.py", "old": "if n < lo:", "new": "if n <= lo:"}',
        'Run the visible tests (at most 3 times): {"op": "run_tests"}',
        'Finish by grading your edited file: {"op": "submit", "answer": {}}',
    ]
    return "\n".join(lines)


def _kill_prompt(entry: str, level: int, diffs: list[str]) -> str:
    count = MAX_TESTS[level]
    target = "the mutant" if len(diffs) == 1 else f"each of the {len(diffs)} mutants"
    lines = [
        f"original.py holds `{entry}`, which is correct. "
        + ("mutant.diff shows a mutant: " if len(diffs) == 1 else f"{', '.join(diffs)} each show a mutant: ")
        + "the same function with one small change.",
        f"Write test cases that kill {target}: the original and the mutant must return different results "
        "for your input, and expected must be exactly what the ORIGINAL returns.",
        f"Submit at most {count} test case{'s' if count > 1 else ''}. Each test is "
        '{"args": [positional arguments as JSON], "expected": <original return value as JSON>}. '
        'If the original raises an exception, use "expected": {"raises": "ValueError"}. '
        f"Keep each test under {MAX_ARGS_BYTES} characters of JSON.",
        "Scoring per mutant: half for an input that tells them apart, half more when expected is also exact.",
        "",
        REPLY_FORMAT,
        'First action: {"op": "read", "path": "*"}',
        'Final action, for example: {"op": "submit", "answer": {"tests": [{"args": [3, 7], "expected": 10}]}}',
    ]
    return "\n".join(lines)


def _candidate_sites(source: str, rng: random.Random) -> list[mutation.Site]:
    sites = mutation.enumerate_sites(source)
    rng.shuffle(sites)
    return sites


def _make_bugfix(task_id: str, level: int, seed: int, rng: random.Random, pool: tuple[str, ...]) -> Task:
    for name in rng.sample(sorted(pool), len(pool)):
        item = corpus.ENTRIES[name]
        original = mutation.rename_locals(item.function_source, random.Random(rng.random()))
        inputs = corpus.gen_inputs(name, rng.randrange(2**31), HIDDEN_INPUTS)
        graded = {json.dumps(a, sort_keys=True) for a in inputs}
        display = [a for a in corpus.gen_inputs(name, rng.randrange(2**31), HIDDEN_INPUTS)
                   if json.dumps(a, sort_keys=True) not in graded]  # shown, never graded
        reference = _outputs(original, item.entry, inputs)
        if not _usable(reference) or not display:
            continue
        chosen, buggy, killed, subtle = [], original, [], 0
        for site in _candidate_sites(original, rng)[:24]:
            try:
                trial = mutation.apply(original, [site] + chosen)
            except ValueError:
                continue  # overlaps an already chosen fault
            alone = _outputs(mutation.apply(original, [site]), item.entry, inputs)
            single = _killing(reference, alone)
            if not _usable(alone) or not single:
                continue  # loops, crashes, or is equivalent on the hidden inputs: never a fault
            is_subtle = len(single) <= SUBTLE_RATE * len(inputs)
            if level == 1 and len({json.dumps(inputs[i], sort_keys=True) for i in single}) < MIN_REVEALING_D1:
                continue  # special-casing the one reported input must not pass every hidden test
            if not is_subtle and MIN_SUBTLE[level] - subtle >= FAULTS[level] - len(chosen):
                continue  # the remaining slots are reserved for subtle faults
            combined = _outputs(trial, item.entry, inputs)
            kills = _killing(reference, combined)
            if not _usable(combined) or not kills:
                continue
            if chosen and not _subsets_all_fail(original, chosen + [site], item.entry, inputs, reference):
                continue  # some faults cancel out: a partial fix would pass
            chosen, buggy, killed, subtle = chosen + [site], trial, kills, subtle + is_subtle
            if len(chosen) == FAULTS[level]:
                break
        if len(chosen) != FAULTS[level]:
            continue
        shown_ref = _outputs(original, item.entry, display)
        shown_bug = _outputs(buggy, item.entry, display)
        if not (_usable(shown_ref) and _usable(shown_bug)):
            continue
        revealing = _killing(shown_ref, shown_bug)
        passing = [i for i in range(len(display)) if i not in revealing]
        if not passing or (level == 1 and not revealing):
            continue
        visible_idx = rng.sample(passing, min(3, len(passing)))
        if level == 1:
            report = (f"A user reports a wrong result for {_call_text(item.entry, display[revealing[0]])} "
                      "(the correct value is not given; derive it from the docstring). The visible tests all pass.")
        else:
            report = ("Hidden tests fail but the visible tests all pass, so no failing input is known. "
                      "Compare the code with the docstring line by line to find every fault.")
        visible = [{"args": display[i], "expected": shown_ref[i]} for i in visible_idx]
        shown = [f"  {_call_text(item.entry, v['args'])} == {_result_text(v['expected'])}" for v in visible]
        return Task(
            task_id=task_id, family="qe_bugfix", difficulty=level, seed=seed,
            prompt=_bugfix_prompt(item.entry, shown, report),
            files={MODULE: buggy},
            private={
                "corpus": name, "entry": item.entry, "original": original, "buggy": buggy,
                "sites": [s.description for s in chosen], "inputs": inputs, "reference": reference,
                "visible": visible, "report_args": display[revealing[0]] if level == 1 else None,
            },
        )
    raise RuntimeError(f"no usable bugfix task for {task_id} seed {seed}")


def _kill_rate(reference: list, outputs: list) -> float:
    return len(_killing(reference, outputs)) / len(reference)


def _make_mutant_kill(task_id: str, level: int, seed: int, rng: random.Random, pool: tuple[str, ...]) -> Task:
    wanted = 3 if level == 3 else 1
    for name in rng.sample(sorted(pool), len(pool)):
        item = corpus.ENTRIES[name]
        original = mutation.rename_locals(item.function_source, random.Random(rng.random()))
        inputs = corpus.gen_inputs(name, rng.randrange(2**31), HIDDEN_INPUTS)
        reference = _outputs(original, item.entry, inputs)
        if not _usable(reference):
            continue
        mutants = []
        for site in _candidate_sites(original, rng)[:24]:
            mutant = mutation.apply(original, [site])
            outputs = _outputs(mutant, item.entry, inputs)
            if not _usable(outputs):
                continue  # loops or crashes: its kill rate cannot be measured reliably
            rate = _kill_rate(reference, outputs)
            if rate == 0:
                continue  # possibly equivalent: never shown
            if level == 1 and rate < 0.25:
                continue
            if level in (2, 3) and rate > SUBTLE_RATE:
                continue
            mutants.append({"source": mutant, "site": site.description, "random_kill_rate": rate})
            if len(mutants) == wanted:
                break
        if len(mutants) < wanted:
            continue
        files = {"original.py": original}
        names = ["mutant.diff"] if wanted == 1 else [f"mutant_{i + 1}.diff" for i in range(wanted)]
        for fname, m in zip(names, mutants):
            files[fname] = mutation.unified_diff(original, m["source"], "original.py")
        return Task(
            task_id=task_id, family="qe_mutant_kill", difficulty=level, seed=seed,
            prompt=_kill_prompt(item.entry, level, names),
            files=files,
            private={"corpus": name, "entry": item.entry, "original": original, "mutants": mutants,
                     "inputs": inputs, "reference": reference},
        )
    raise RuntimeError(f"no usable mutant_kill task for {task_id} seed {seed}")


def make_task(task_id: str, seed: int, *, pool: tuple[str, ...] | None = None) -> Task:
    """Build the task for (task_id, seed).

    ``pool`` restricts which corpus functions may be drawn. It exists for
    per-function calibration and is never reachable from an episode: the
    server constructs environments without it.
    """
    family, level = parse_task_id(task_id)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64 bit integer")
    if pool is None:
        pool = tuple(corpus.ENTRIES)
    elif not pool or any(name not in corpus.ENTRIES for name in pool):
        raise ValueError("pool must name corpus entries")
    rng = random.Random(f"{task_id}:{seed}")
    if family == "qe_bugfix":
        return _make_bugfix(task_id, level, seed, rng, tuple(pool))
    return _make_mutant_kill(task_id, level, seed, rng, tuple(pool))


def run_visible(task: Task, source: str) -> str:
    """Visible-test report for the bugfix working copy (shown to the model)."""
    p = task.private
    calls = [v["args"] for v in p["visible"]]
    results = _outputs(source, p["entry"], calls)
    if results and all(r.kind == "rejected" for r in results):
        return "REJECTED: " + "; ".join(sandbox.static_gate(source))[:400]
    lines = []
    for v, got in zip(p["visible"], results):
        ok = sandbox.compare(v["expected"], got)
        lines.append(f"{'PASS' if ok else 'FAIL'} {_call_text(p['entry'], v['args'])}"
                     + ("" if ok else f" -> got {_result_text(got)}, expected {_result_text(v['expected'])}"))
    passed = sum(line.startswith("PASS") for line in lines)
    return f"{passed}/{len(lines)} visible tests passed\n" + "\n".join(lines)


def grade_bugfix(task: Task, source: str) -> float:
    p = task.private
    if source.strip() == p["buggy"].strip():
        return 0.0
    results = _outputs(source, p["entry"], p["inputs"])
    return 1.0 if len(results) == len(p["reference"]) and all(
        sandbox.compare(r, o) for r, o in zip(p["reference"], results)) else 0.0


def _as_actual_types(expected: Any, actual: Any) -> Any:
    """JSON writes 2.0 as 2: read an int as a float wherever the original returned a float."""
    if type(expected) is int and type(actual) is float:
        return float(expected)
    if type(expected) is list and type(actual) is list and len(expected) == len(actual):
        return [_as_actual_types(e, a) for e, a in zip(expected, actual)]
    if type(expected) is dict and type(actual) is dict:
        return {k: _as_actual_types(v, actual[k]) if k in actual else v for k, v in expected.items()}
    return expected


def _expected_matches(expected: Any, actual: sandbox.CallResult) -> bool:
    """Exact match against what the original actually did; never raises."""
    try:
        if actual.kind == "exc":
            return type(expected) is dict and set(expected) == {"raises"} and expected["raises"] == actual.value
        if actual.kind != "ok":
            return False
        return sandbox.compare(sandbox.CallResult(kind="ok", value=_as_actual_types(expected, actual.value)), actual)
    except (OverflowError, ValueError, TypeError, RecursionError):
        return False


def grade_mutant_kill(task: Task, answer: Any) -> float:
    p = task.private
    tests = answer.get("tests") if isinstance(answer, dict) else None
    if not isinstance(tests, list) or not tests or len(tests) > MAX_TESTS[task.difficulty]:
        return 0.0
    calls, expecteds = [], []
    for test in tests:
        if not isinstance(test, dict) or set(test) != {"args", "expected"} or not isinstance(test["args"], list):
            return 0.0
        if len(json.dumps(test, separators=(",", ":"))) > MAX_ARGS_BYTES:
            return 0.0
        calls.append(test["args"])
        expecteds.append(test["expected"])
    # One batch per test: a trap ends a batch and skips the calls after it, so tests sharing a batch
    # would score differently depending on their order.
    reference = [r for call in calls for r in _outputs(p["original"], p["entry"], [call])]
    scores = []
    for m in p["mutants"]:
        outputs = [r for call in calls for r in _outputs(m["source"], p["entry"], [call])]
        best = 0.0
        for i in _killing(reference, outputs):  # only settled, differing results count as a kill
            best = max(best, 1.0 if _expected_matches(expecteds[i], reference[i]) else 0.5)
        scores.append(best)
    return sum(scores) / len(scores)
