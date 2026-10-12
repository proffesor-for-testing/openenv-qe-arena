"""The real-bug tier: upstream pre-fix faults served by qe_bugfix through the same reset/step contract."""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import solve_actions  # noqa: E402
from qe_arena import corpus, mutation, sandbox, tasks  # noqa: E402
from qe_arena.environment import QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, make_task  # noqa: E402

PRE_FIX_RE = re.compile(r"^ruvnet/RuVector@3e1d1886c50f2db817664648b95fab18e3e4c34b:crates/[\w./-]+\.rs$")
FIXED_RE = re.compile(r"^ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/[\w./-]+\.rs$")
REAL_FAULTS = [(name, fault) for name in corpus.REAL_NAMES for fault in corpus.ENTRIES[name].real_faults]
LEVELS = (1, 2, 3)
REAL_SEEDS = (0, 1, 2, 3)


def _play(task, actions, fault_kind=None):
    env = QEEnvironment(fault_kind=fault_kind)
    obs = env.reset(seed=task.seed, task_id=task.task_id)
    assert obs.prompt == task.prompt  # the environment serves exactly the task built here
    for action in actions:
        obs = env.step(QEAction(**action))
        if obs.done:
            break
    return obs


def test_every_real_bug_entry_has_provenance_for_both_sides():
    assert len(corpus.REAL_NAMES) >= 5
    assert len(REAL_FAULTS) >= 8
    assert not set(corpus.REAL_NAMES) & set(corpus.INJECTED_NAMES)
    ids = [fault.id for _, fault in REAL_FAULTS]
    assert len(ids) == len(set(ids))
    for name in corpus.REAL_NAMES:
        for origin in corpus.ENTRIES[name].origin.split("; "):
            assert FIXED_RE.match(origin), origin
    for name, fault in REAL_FAULTS:
        assert fault.id.startswith(name + "/")
        assert PRE_FIX_RE.match(fault.origin), fault.origin
        assert fault.summary and fault.kind


@pytest.mark.parametrize("name, fault", REAL_FAULTS, ids=[f.id for _, f in REAL_FAULTS])
def test_real_fault_is_an_exact_gated_replacement(name, fault):
    item = corpus.ENTRIES[name]
    assert item.function_source.count(fault.find) == 1
    variant = item.function_source.replace(fault.find, fault.replace)
    assert variant != item.function_source
    assert sandbox.static_gate(variant) == []


@pytest.mark.parametrize("name, fault", REAL_FAULTS, ids=[f.id for _, f in REAL_FAULTS])
def test_real_fault_is_revealed_by_enough_distinct_graded_inputs(name, fault):
    item = corpus.ENTRIES[name]
    variant = item.function_source.replace(fault.find, fault.replace)
    for seed in range(8):
        inputs = corpus.gen_inputs(name, seed, tasks.HIDDEN_INPUTS)
        reference = sandbox.run_calls(item.function_source, item.entry, inputs)
        outputs = sandbox.run_calls(variant, item.entry, inputs)
        assert all(r.kind in ("ok", "exc") for r in reference + outputs)
        revealing = {json.dumps(inputs[i], sort_keys=True) for i in tasks._killing(reference, outputs)}
        assert len(revealing) >= tasks.MIN_REVEALING_D1, (seed, len(revealing))


@pytest.mark.parametrize("name", corpus.REAL_NAMES)
def test_real_sites_survive_renaming(name):
    import random  # noqa: PLC0415

    item = corpus.ENTRIES[name]
    for seed in range(10):
        renaming = mutation.local_renaming(item.function_source, random.Random(seed))
        original = mutation.rename_with(item.function_source, renaming)
        found = tasks._real_sites(item, original, renaming)
        assert len(found) == len(item.real_faults)
        for site, fault in found.items():
            buggy = mutation.apply(original, [site])
            expected = mutation.rename_with(item.function_source.replace(fault.find, fault.replace), renaming)
            assert buggy == expected


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", REAL_SEEDS)
def test_real_tier_task_contract(level, seed):
    task = make_task(f"qe_bugfix-d{level}", seed, fault_kind="real")
    p = task.private
    assert p["fault_kind"] == "real" and p["corpus"] in corpus.REAL_NAMES
    assert len(p["fault_sites"]) == tasks.FAULTS[level]
    assert any(f["kind"] == "real" for f in p["faults"])
    for record in p["faults"]:
        assert set(record) >= {"kind", "description"}
        if record["kind"] == "real":
            assert PRE_FIX_RE.match(record["origin"])
    assert mutation.apply(p["original"], p["fault_sites"]) == p["buggy"] == task.files[MODULE]
    buggy_out = sandbox.run_calls(p["buggy"], p["entry"], p["inputs"])
    assert tasks._killing(p["reference"], buggy_out)
    for v in p["visible"]:
        [got] = sandbox.run_calls(p["buggy"], p["entry"], [v["args"]])
        assert sandbox.compare(v["expected"], got)
    # Nothing about the fault source reaches the model.
    assert not re.search(r"\breal\b", task.prompt, re.I) and "RuVector" not in json.dumps(task.files)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", REAL_SEEDS)
def test_real_tier_oracle_earns_one_and_doing_nothing_earns_zero(level, seed):
    task = make_task(f"qe_bugfix-d{level}", seed, fault_kind="real")
    assert _play(task, solve_actions(task), fault_kind="real").reward == 1.0
    assert _play(task, [{"op": "submit", "answer": {}}], fault_kind="real").reward == 0.0


@pytest.mark.parametrize("level", (2, 3))
def test_real_tier_partial_fix_earns_zero(level):
    task = make_task(f"qe_bugfix-d{level}", 5, fault_kind="real")
    p = task.private
    for keep in p["fault_sites"]:
        partial = mutation.apply(p["original"], [keep])
        assert tasks.grade_bugfix(task, partial) == 0.0


def test_default_bugfix_mixes_real_and_injected_faults():
    kinds = Counter(make_task("qe_bugfix-d2", seed).private["fault_kind"] for seed in range(100))
    assert set(kinds) == {"real", "injected"}
    assert 0.05 <= kinds["real"] / 100 <= 2 * tasks.REAL_SHARE, kinds  # at most REAL_SHARE, less on fallbacks


def test_injected_tier_and_mutant_kill_never_draw_real_bug_entries():
    for seed in range(12):
        assert make_task("qe_bugfix-d2", seed, fault_kind="injected").private["corpus"] in corpus.INJECTED_NAMES
        assert make_task("qe_mutant_kill-d2", seed).private["corpus"] in corpus.INJECTED_NAMES


def test_fault_kind_is_validated_and_unreachable_from_reset_kwargs():
    with pytest.raises(ValueError):
        make_task("qe_bugfix-d1", 0, fault_kind="natural")
    with pytest.raises(ValueError):
        make_task("qe_mutant_kill-d1", 0, fault_kind="real")
    with pytest.raises(ValueError):
        make_task("qe_bugfix-d1", 0, pool=tuple(corpus.INJECTED_NAMES[:3]), fault_kind="real")
    env = QEEnvironment()
    obs = env.reset(seed=3, task_id="qe_bugfix-d1", fault_kind="real")  # an unknown kwarg changes nothing
    assert obs.prompt == make_task("qe_bugfix-d1", 3).prompt


@pytest.mark.parametrize("task_id", tasks.TASK_IDS)
def test_oracle_earns_one_on_many_seeds_for_every_cell(task_id):
    """Across the default mix (real and injected bug-fix faults), the reference solution always pays 1.0."""
    kinds = Counter()
    for seed in range(100, 116):
        task = make_task(task_id, seed)
        kinds[task.private.get("fault_kind")] += 1
        env = QEEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        for action in solve_actions(task):
            obs = env.step(QEAction(**action))
            if obs.done:
                break
        assert obs.done and obs.reward == 1.0, (task_id, seed)
    if task_id.startswith("qe_bugfix"):
        assert kinds["real"] >= 1 and kinds["injected"] >= 1, kinds


# --------------------------------------------------------------------------- spec sufficiency

# Implementations written from each real entry's docstring alone, one per plausible literal reading.
# If every reading equals the reference, no served fault can be "what the docstring says": a fault
# differs from the reference on some graded input, so it differs from every reading there too.
# (Review of 65ac734, MAJOR-1: "an int" admitted bools, and condition_drop of the bool check matched it.)

def _find_stop(text, stop_sequences):
    hits = [text.find(s) for s in stop_sequences if s]
    hits = [h for h in hits if h >= 0]
    return min(hits) if hits else None


def _streamable_len(text, holdback):
    if holdback < 0:
        raise ValueError
    body = text.rstrip("�")
    data, end = text.encode("utf-8"), max(len(body.encode("utf-8")) - holdback, 0)
    while 0 < end < len(data) and (data[end] & 0xC0) == 0x80:
        end -= 1
    return end


def _gguf_loader(metadata, bool_is_int):
    parts = metadata.get("split.count")
    is_int = isinstance(parts, int) if bool_is_int else type(parts) is int
    if is_int and parts > 1:
        raise ValueError
    arch = metadata.get("general.architecture")
    name = arch.lower() if isinstance(arch, str) else ""
    table = {"": "llama", "llama": "llama", "mistral": "llama", "qwen2": "qwen2", "qwen3": "qwen3"}
    if name not in table:
        raise ValueError
    return table[name]


def _gguf_special_ids(metadata, vocab, bool_is_int=False):
    def declared(x):
        v = metadata.get("tokenizer.ggml." + x + "_token_id")
        is_int = isinstance(v, int) if bool_is_int else type(v) is int
        return v if is_int and 0 <= v < len(vocab) else None
    eos = declared("eos")
    if eos is None:
        eos = next((vocab[t] for t in ["</s>", "<end_of_text>", "<endoftext>", "<eot_id>"] if t in vocab), None)
    stop = [vocab[t] for t in ["<im_end>", "<endoftext>", "<eot_id>", "<end_of_text>", "</s>"] if t in vocab]
    for x in ("eos", "eot", "eom"):
        v = declared(x)
        if v is not None and v not in stop:
            stop.append(v)
    return {"eos": eos, "stop": stop}


def _chat_completion_result(cause, prompt, completion, p, c):
    if p < 0 or c < 0:
        raise ValueError
    reason = {"length": "length", "stop": "stop", "end_of_sequence": "stop", "cancelled": "stop"}.get(cause)
    if reason is None:
        raise ValueError
    return {"finish_reason": reason, "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}


SPEC_READINGS = {
    "find_stop": [_find_stop],
    "streamable_len": [_streamable_len],
    "gguf_loader": [lambda m: _gguf_loader(m, True), lambda m: _gguf_loader(m, False)],
    "gguf_special_ids": [_gguf_special_ids],
    "chat_completion_result": [_chat_completion_result],
}


def _entry_doc(name):
    item = corpus.ENTRIES[name]
    entry = next(n for n in ast.parse(item.function_source).body if isinstance(n, ast.FunctionDef) and n.name == item.entry)
    return ast.get_docstring(entry)


def readings(name):
    """The readings the current docstring permits. A bare "an int" also admits bools in Python."""
    found = list(SPEC_READINGS[name])
    if name == "gguf_special_ids" and "(not a bool)" not in _entry_doc(name):
        found.append(lambda m, v: _gguf_special_ids(m, v, bool_is_int=True))
    return found


def _run(fn, args):
    try:
        return "ok", json.dumps(fn(*json.loads(json.dumps(args))), sort_keys=True)
    except Exception as exc:  # noqa: BLE001 - compare by exception type, like the grader
        return "exc", type(exc).__name__


def test_every_real_entry_has_spec_readings():
    assert set(SPEC_READINGS) == set(corpus.REAL_NAMES)


@pytest.mark.parametrize("name", corpus.REAL_NAMES)
def test_docstring_readings_equal_the_reference(name):
    for seed in range(100):
        for args in corpus.gen_inputs(name, seed, tasks.HIDDEN_INPUTS):
            reference = _run(lambda *a: corpus.call_reference(name, list(a)), args)
            for reading in readings(name):
                assert _run(reading, args) == reference, (name, args)


@pytest.mark.parametrize("level", LEVELS)
def test_no_served_real_fault_matches_a_literal_reading(level):
    for seed in range(6):
        p = make_task(f"qe_bugfix-d{level}", seed, fault_kind="real").private
        for site in p["fault_sites"]:
            outputs = sandbox.run_calls(mutation.apply(p["original"], [site]), p["entry"], p["inputs"])
            for reading in readings(p["corpus"]):
                literal = [_run(reading, args) for args in p["inputs"]]
                got = [("ok", json.dumps(o.value, sort_keys=True)) if o.kind == "ok" else ("exc", o.value)
                       for o in outputs]
                assert got != literal, (p["corpus"], site.description)


def test_gguf_special_ids_docstring_excludes_bools():
    assert "an int (not a bool)" in _entry_doc("gguf_special_ids")


def test_bool_admitting_reading_differs_from_the_reference_only_on_bools():
    """Negative control: the reading of the old docstring ("an int") is wrong exactly on bool ids."""
    lenient = lambda m, v: _gguf_special_ids(m, v, bool_is_int=True)  # noqa: E731
    differing = []
    for seed in range(50):
        for metadata, vocab in corpus.gen_inputs("gguf_special_ids", seed, tasks.HIDDEN_INPUTS):
            reference = _run(lambda *a: corpus.call_reference("gguf_special_ids", list(a)), [metadata, vocab])
            if _run(lenient, [metadata, vocab]) != reference:
                differing.append(metadata)
    assert differing, "the generated inputs must contain bool ids, or MAJOR-1 cannot be caught"
    assert all(any(type(v) is bool for v in m.values()) for m in differing)


# --------------------------------------------------------------------------- fallbacks and determinism

def test_real_draw_falls_back_to_injected_when_no_real_fault_fits(monkeypatch):
    monkeypatch.setattr(tasks, "_real_sites", lambda item, original, renaming: {})
    seed = next(s for s in range(200) if __import__("random").Random(f"qe_bugfix-d1:{s}").random() < tasks.REAL_SHARE)
    task = make_task("qe_bugfix-d1", seed)
    assert task.private["fault_kind"] == "injected" and task.private["corpus"] in corpus.INJECTED_NAMES
    with pytest.raises(RuntimeError):
        make_task("qe_bugfix-d1", seed, fault_kind="real")


def test_real_sites_skip_a_fault_whose_pre_fix_code_collides_with_the_renaming():
    item = corpus.CorpusEntry(
        name="demo", entry="f", origin="", tags=(), source="", summary="",
        function_source="def f(a):\n    b = a + 1\n    return b\n",
        real_faults=(corpus.RealFault("demo/x", "k", "s", "o", "    return b\n", "    c = 2\n    return b + c\n"),
                     corpus.RealFault("demo/y", "k", "s", "o", "    return b\n", "    return b - 1\n")))
    renaming = {"b": "c"}  # fine for the fixed code, but "c" is a name in demo/x's pre-fix code
    original = mutation.rename_with(item.function_source, renaming)
    found = tasks._real_sites(item, original, renaming)
    assert [fault.id for fault in found.values()] == ["demo/y"]


def test_task_generation_is_identical_across_processes_and_hash_seeds():
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    code = (
        "import hashlib, json, sys\n"
        "from qe_arena.tasks import make_task\n"
        "out = []\n"
        "for tid, seed, kind in [('qe_bugfix-d1', 3, None), ('qe_bugfix-d2', 5, 'real'), ('qe_bugfix-d3', 7, None),"
        " ('qe_bugfix-d3', 2, 'real'), ('qe_mutant_kill-d3', 4, None)]:\n"
        "    t = make_task(tid, seed, fault_kind=kind)\n"
        "    blob = json.dumps([t.prompt, t.files, t.private.get('sites'), t.private['inputs']], sort_keys=True)\n"
        "    out.append(hashlib.sha256(blob.encode()).hexdigest())\n"
        "print(json.dumps(out))\n"
    )
    root = Path(__file__).resolve().parents[1]
    results = set()
    for hash_seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        done = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True,
                              timeout=600, check=True)
        results.add(done.stdout.strip())
    assert len(results) == 1, results
