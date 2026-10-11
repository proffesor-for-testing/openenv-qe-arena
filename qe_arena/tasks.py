"""Task generation and grading for the quality-engineering families.

A task is fully determined by ``(task_id, seed)``. ``private`` holds verifier
material (reference outputs, mutants, hidden inputs) and MUST NOT be sent in an
observation. Every reference output comes from running our own corpus code in
the sandbox, so the oracle and the candidate share one execution semantics.
"""

from __future__ import annotations

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


def _result_text(result: sandbox.CallResult) -> str:
    if result.kind == "ok":
        return _show(result.value)
    if result.kind == "exc":
        return f"raises {result.value}"
    return result.kind


def _outputs(source: str, entry: str, calls: list[list]) -> list[sandbox.CallResult]:
    return sandbox.run_calls(source, entry, calls)


def _killing(reference: list, outputs: list) -> list[int]:
    return [i for i, (r, o) in enumerate(zip(reference, outputs)) if not sandbox.compare(r, o)]


def _usable(reference: list) -> bool:
    return all(r.kind in ("ok", "exc") for r in reference)


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
        'If the original raises an exception, use "expected": {"raises": "ValueError"}.',
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


def _make_bugfix(task_id: str, level: int, seed: int, rng: random.Random) -> Task:
    for name in rng.sample(sorted(corpus.ENTRIES), len(corpus.ENTRIES)):
        item = corpus.ENTRIES[name]
        original = mutation.rename_locals(item.function_source, random.Random(rng.random()))
        inputs = corpus.gen_inputs(name, rng.randrange(2**31), HIDDEN_INPUTS)
        reference = _outputs(original, item.entry, inputs)
        if not _usable(reference):
            continue
        chosen, buggy, killed, subtle = [], original, [], 0
        for site in _candidate_sites(original, rng)[:24]:
            try:
                trial = mutation.apply(original, [site] + chosen)
            except ValueError:
                continue  # overlaps an already chosen fault
            single = _killing(reference, _outputs(mutation.apply(original, [site]), item.entry, inputs))
            if not single:
                continue  # equivalent on the hidden inputs: never a fault
            is_subtle = len(single) <= SUBTLE_RATE * len(inputs)
            if not is_subtle and MIN_SUBTLE[level] - subtle >= FAULTS[level] - len(chosen):
                continue  # the remaining slots are reserved for subtle faults
            kills = _killing(reference, _outputs(trial, item.entry, inputs))
            if not kills:
                continue  # faults that cancel each other out
            chosen, buggy, killed, subtle = chosen + [site], trial, kills, subtle + is_subtle
            if len(chosen) == FAULTS[level]:
                break
        if len(chosen) != FAULTS[level]:
            continue
        passing = [i for i in range(len(inputs)) if i not in killed]
        visible_idx = rng.sample(passing, min(3, len(passing)))
        if level == 1:
            report = (f"A user reports a wrong result for {_call_text(item.entry, inputs[killed[0]])} "
                      "(the correct value is not given; derive it from the docstring). The visible tests all pass.")
        else:
            report = ("Hidden tests fail but the visible tests all pass, so no failing input is known. "
                      "Compare the code with the docstring line by line to find every fault.")
        visible = [{"args": inputs[i], "expected": reference[i]} for i in visible_idx]
        shown = [f"  {_call_text(item.entry, v['args'])} == {_result_text(v['expected'])}" for v in visible]
        return Task(
            task_id=task_id, family="qe_bugfix", difficulty=level, seed=seed,
            prompt=_bugfix_prompt(item.entry, shown, report),
            files={MODULE: buggy},
            private={
                "corpus": name, "entry": item.entry, "original": original, "buggy": buggy,
                "sites": [s.description for s in chosen], "inputs": inputs, "reference": reference,
                "visible": visible,
            },
        )
    raise RuntimeError(f"no usable bugfix task for {task_id} seed {seed}")


def _kill_rate(reference: list, outputs: list) -> float:
    return len(_killing(reference, outputs)) / len(reference)


def _make_mutant_kill(task_id: str, level: int, seed: int, rng: random.Random) -> Task:
    wanted = 3 if level == 3 else 1
    for name in rng.sample(sorted(corpus.ENTRIES), len(corpus.ENTRIES)):
        item = corpus.ENTRIES[name]
        original = mutation.rename_locals(item.function_source, random.Random(rng.random()))
        inputs = corpus.gen_inputs(name, rng.randrange(2**31), HIDDEN_INPUTS)
        reference = _outputs(original, item.entry, inputs)
        if not _usable(reference):
            continue
        mutants = []
        for site in _candidate_sites(original, rng)[:24]:
            mutant = mutation.apply(original, [site])
            rate = _kill_rate(reference, _outputs(mutant, item.entry, inputs))
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


def make_task(task_id: str, seed: int) -> Task:
    family, level = parse_task_id(task_id)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64 bit integer")
    rng = random.Random(f"{task_id}:{seed}")
    if family == "qe_bugfix":
        return _make_bugfix(task_id, level, seed, rng)
    return _make_mutant_kill(task_id, level, seed, rng)


def run_visible(task: Task, source: str) -> str:
    """Visible-test report for the bugfix working copy (shown to the model)."""
    p = task.private
    calls = [v["args"] for v in p["visible"]]
    results = sandbox.run_calls(source, p["entry"], calls)
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
    results = sandbox.run_calls(source, p["entry"], p["inputs"])
    return 1.0 if len(results) == len(p["reference"]) and all(
        sandbox.compare(r, o) for r, o in zip(p["reference"], results)) else 0.0


def _expected_matches(expected: Any, actual: sandbox.CallResult) -> bool:
    if isinstance(expected, dict) and set(expected) == {"raises"}:
        return actual.kind == "exc" and actual.value == expected["raises"]
    if actual.kind == "ok" and isinstance(actual.value, float) and type(expected) is int:
        expected = float(expected)  # JSON writes 2.0 as 2; the model should not lose for that
    return sandbox.compare(sandbox.CallResult(kind="ok", value=expected), actual)


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
    reference = sandbox.run_calls(p["original"], p["entry"], calls)
    scores = []
    for m in p["mutants"]:
        outputs = sandbox.run_calls(m["source"], p["entry"], calls)
        best = 0.0
        for ref, out, exp in zip(reference, outputs, expecteds):
            if ref.kind not in ("ok", "exc") or sandbox.compare(ref, out):
                continue
            best = max(best, 1.0 if _expected_matches(exp, ref) else 0.5)
        scores.append(best)
    return sum(scores) / len(scores)
