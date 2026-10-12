"""The real-bug tier: upstream pre-fix faults served by qe_bugfix through the same reset/step contract."""

from __future__ import annotations

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
        assert FIXED_RE.match(corpus.ENTRIES[name].origin), corpus.ENTRIES[name].origin
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
    kinds = Counter(make_task("qe_bugfix-d1", seed).private["fault_kind"] for seed in range(40))
    assert set(kinds) == {"real", "injected"}
    assert 0.1 <= kinds["real"] / 40 <= 0.45, kinds


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
