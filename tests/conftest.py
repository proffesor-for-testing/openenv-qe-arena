"""Split the suite into a fast pre-merge gate and a slow post-merge run.

`pytest -m "not slow"` is the gate CI runs before building the image; `pytest -m slow` runs on main
after the merge, and a submission waits for it. A test is slow when it uses a fixture that builds
hundreds of tasks, or when one of its cases took 10 s or more in a timed run on ruos-dev (8 xdist
workers, 2 WASI workers each, 2026-10-12). Re-time with `--junitxml` when the suite changes shape.
"""

from __future__ import annotations

import os
import zlib

import pytest

HEAVY_FIXTURES = frozenset({"many", "d3_by_cause"})

SLOW = frozenset({
    # d4, timed on a loaded host (about 4x the idle times above), so the cut there was 40 s
    "test_bugfix_d4.py::test_a_full_rewrite_is_refused_unapplied_and_free",
    "test_bugfix_d4.py::test_export_rows_keep_the_base_cells_byte_identical",
    "test_bugfix_d4.py::test_oracle_passes_the_gate_on_a_fixed_sample",
    "test_bugfix_d4.py::test_refused_edits_are_free_up_to_a_cap_so_episodes_stay_bounded",
    "test_d4_attacks.py::test_attacks_played_as_edits_are_refused",
    "test_d4_attacks.py::test_every_attack_breaks_the_gate_and_grades_zero",
    "test_d4_attacks.py::test_exits_that_end_the_block_make_the_old_body_dead",
    "test_d4_attacks.py::test_guarded_rewrites_behind_any_fixed_test_never_pay",
    "test_d4_attacks.py::test_oracle_earns_one_on_300_seeds_within_budget_gate_and_steps",
    "test_d4_attacks.py::test_oracle_earns_one_on_1000_natural_seeds",
    "test_domains.py::test_flaky_oracle_earns_one_and_doing_nothing_zero_on_a_domain_cell",
    "test_domains.py::test_oracle_earns_one_on_many_seeds_for_every_domain_cell",
    "test_environment.py::test_bugfix_leaving_any_single_fault_unfixed_earns_zero",
    "test_environment.py::test_oracle_solution_earns_exactly_one",
    "test_environment.py::test_prompt_states_the_reply_format_with_concrete_actions",
    "test_environment.py::test_same_seed_reproduces_the_task_and_different_seeds_vary",
    "test_flaky.py::test_generation_is_deterministic_per_seed",
    "test_flaky_audit.py::test_a_low_rate_flake_band_exists_and_is_still_observed",
    "test_flaky_audit.py::test_drawing_from_a_huge_range_keeps_the_defect",
    "test_flaky_audit.py::test_emptying_the_polluter_never_earns_the_repair",
    "test_flaky_audit.py::test_m1_run_once_through_a_mutable_default_is_refused",
    "test_flaky_audit.py::test_m2_a_probe_draw_before_the_flaky_draw_does_not_dodge_the_sweep",
    "test_flaky_audit.py::test_m2_no_constant_identifies_sweep_runs_across_tasks",
    "test_flaky_audit.py::test_m2_the_auditors_sweep_detecting_hacks_score_half",
    "test_flaky_audit.py::test_m3_an_order_insensitive_pool_repair_earns_full_credit",
    "test_flaky_audit.py::test_patching_env_frac_never_earns_the_repair",
    "test_flaky_audit.py::test_patching_the_env_class_never_earns_the_repair",
    "test_flaky_audit.py::test_shadowing_a_builtin_above_the_harness_never_earns_the_repair",
    "test_flaky_audit.py::test_the_syntax_only_classifier_scores_near_the_null_baseline",
    "test_flaky_audit.py::test_widening_the_clock_bucket_keeps_the_defect",
    "test_flywheel_search.py::test_iid_arms_are_rarely_accepted_at_the_default_plan_size",
    "test_manifest.py::test_every_tampered_field_fails_the_signature",
    "test_manifest.py::test_recompute_equals_the_manifest_for_several_tasks",
    "test_mutation.py::test_mutants_change_behaviour_on_sample",
    "test_mutation.py::test_realistic_sites_on_the_corpus_are_gated_and_never_unbound",
    "test_real_bugs.py::test_default_bugfix_mixes_real_and_injected_faults",
    "test_real_bugs.py::test_injected_tier_and_mutant_kill_never_draw_real_bug_entries",
    "test_real_bugs.py::test_no_served_real_fault_matches_a_literal_reading",
    "test_real_bugs.py::test_oracle_earns_one_on_many_seeds_for_every_cell",
    "test_real_bugs.py::test_real_tier_oracle_earns_one_and_doing_nothing_earns_zero",
    "test_real_bugs.py::test_task_generation_is_identical_across_processes_and_hash_seeds",
    "test_review_findings.py::test_oracle_earns_one_on_many_seeds",
    "test_tasks_mutation_gaps.py::test_buggy_file_settles_on_every_hidden_and_shown_input",
})


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long statistical or many-seed test; runs after merge, not in the gate")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    seen, files = set(), set()
    for item in items:
        key = f"{item.path.name}::{getattr(item, 'originalname', item.name)}"
        seen.add(key)
        files.add(item.path.name)
        if key in SLOW or HEAVY_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.slow)
    # QE_ARENA_SHARD=i/n keeps the tests whose node id hashes to shard i (crc32, stable across runs and
    # hosts), so CI can split the gate over n runners. Unset: every test runs.
    shard = os.environ.get("QE_ARENA_SHARD")
    if shard:
        index, count = (int(x) for x in shard.split("/"))
        if not 0 <= index < count:
            raise pytest.UsageError(f"QE_ARENA_SHARD={shard}: need 0 <= i < n")
        keep, drop = [], []
        for item in items:
            (keep if zlib.crc32(item.nodeid.encode()) % count == index else drop).append(item)
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
    # A renamed slow test would silently move into the gate; fail collection instead. Only checked for
    # files that were collected in full, so `-k`, `-m` and node-id selection still work.
    if not config.getoption("keyword") and not any("::" in a for a in config.args):
        stale = sorted(k for k in SLOW if k.split("::")[0] in files and k not in seen)
        if stale:
            raise pytest.UsageError(f"tests/conftest.py SLOW names tests that no longer exist: {stale}")
