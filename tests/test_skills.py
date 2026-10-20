"""qe_skill verifier tests: the oracle solves every seed, every listed wrong method fails, nothing leaks.

Each family's solver works from the visible files alone, so "the oracle earns 1.0" also proves the workspace holds
everything the task needs. Generation retries until every trap that applies at the level fails, so the trap tests
assert every seed, not a share.
"""

from __future__ import annotations

import json

import pytest

from qe_arena import shell, skills
from qe_arena.terminal_env import MAX_STEPS, TERMINAL_TASK_IDS, TerminalAction, TerminalEnvironment, instruction

SEEDS = range(200)
CELLS = skills.SKILL_TASK_IDS


@pytest.fixture(autouse=True)
def unsandboxed(monkeypatch, tmp_path):
    monkeypatch.setenv(shell.ENV_FLAG, "1")
    monkeypatch.setenv(shell.ROOT_ENV, str(tmp_path / "ws"))
    shell._reset_mode_for_tests()
    yield
    shell._reset_mode_for_tests()


def test_every_domain_cell_is_served_by_the_terminal_and_appended_after_the_code_cells():
    assert CELLS == [f"qe_skill-{slug}-d{level}" for slug in ("science", "math", "media") for level in (1, 2, 3)]
    assert TERMINAL_TASK_IDS[-len(CELLS):] == CELLS


@pytest.mark.parametrize("task_id", CELLS)
def test_the_oracle_earns_one_on_every_seed(task_id):
    for seed in SEEDS:
        task = skills.make_skill_task(task_id, seed)
        assert skills.grade(task, skills.solve(task)) == 1.0, (task.kind, seed)


@pytest.mark.parametrize("task_id", CELLS)
def test_every_applicable_trap_scores_zero_on_every_seed(task_id):
    seen = set()
    for seed in SEEDS:
        task = skills.make_skill_task(task_id, seed)
        for trap, levels in task.private["family"].traps.items():
            if task.difficulty in levels:
                seen.add((task.kind, trap))
                assert skills.grade(task, skills.solve(task, trap)) == 0.0, (task.kind, trap, seed)
    assert seen, "a cell with no traps tests nothing"


@pytest.mark.parametrize("task_id", CELLS)
def test_empty_copied_and_malformed_outputs_score_zero(task_id):
    for seed in range(40):
        task = skills.make_skill_task(task_id, seed)
        good = skills.solve(task)
        bad = [None, "", " ", "{}", "[]", "null", good[: len(good) // 2], good + "\n{}" if good.startswith("{") else "x",
               "x" * (skills.MAX_OUTPUT_BYTES + 1), *task.files.values()]
        for text in bad:
            assert skills.grade(task, text) == 0.0, (task.kind, seed, (text or "")[:40])


def test_non_finite_numbers_and_booleans_never_pass_a_numeric_check():
    task = next(t for t in (skills.make_skill_task("qe_skill-science-d1", s) for s in SEEDS) if t.kind == "units")
    out = json.loads(skills.solve(task))
    sample = next(iter(out["means"]))
    for value in ("NaN", "Infinity", "true", '"1.0"'):
        text = json.dumps(out).replace(json.dumps({sample: out["means"][sample]})[1:-1],
                                       f'"{sample}": {value}')
        assert skills.grade(task, text) == 0.0, value


def test_numeric_answers_pass_inside_the_stated_rounding_and_fail_outside_it():
    task = next(t for t in (skills.make_skill_task("qe_skill-science-d2", s) for s in SEEDS) if t.kind == "units")
    out = json.loads(skills.solve(task))
    sample = next(iter(out["means"]))
    for delta, expected in ((0.0004, 1.0), (-0.0004, 1.0), (0.002, 0.0)):
        moved = dict(out, means=dict(out["means"], **{sample: out["means"][sample] + delta}))
        assert skills.grade(task, json.dumps(moved)) == expected, delta


def test_fractions_must_be_reduced_strings_and_srt_times_allow_one_millisecond():
    task = next(t for t in (skills.make_skill_task("qe_skill-math-d1", s) for s in SEEDS) if t.kind == "linear")
    out = json.loads(skills.solve(task))
    first = out["x"][0]
    num, _, den = first.partition("/")
    for wrong in (" " + first, "+" + first.lstrip("-"), f"{int(num) * 2}/{int(den or 1) * 2}", first + ".0"):
        assert skills.grade(task, json.dumps(dict(out, x=[wrong, *out["x"][1:]]))) == 0, wrong
    task = next(t for t in (skills.make_skill_task("qe_skill-media-d1", s) for s in SEEDS) if t.kind == "subtitles")
    text = skills.solve(task)
    first = text.split("\n")[1]
    end_ms = int(first[-3:])
    one_off = first[:-3] + f"{(end_ms + 1) % 1000:03d}" if end_ms < 999 else first
    five_off = first[:-3] + f"{(end_ms + 5) % 1000:03d}" if end_ms < 995 else first
    if end_ms < 995:
        assert skills.grade(task, text.replace(first, one_off, 1)) == 1.0
        assert skills.grade(task, text.replace(first, five_off, 1)) == 0.0


def test_same_seed_rebuilds_the_same_task_and_seeds_differ():
    for task_id in CELLS:
        a, b = skills.make_skill_task(task_id, 5), skills.make_skill_task(task_id, 5)
        assert (a.kind, a.files, a.prompt, a.output) == (b.kind, b.files, b.prompt, b.output)
        assert a.files != skills.make_skill_task(task_id, 6).files or a.kind != skills.make_skill_task(task_id, 6).kind


def test_the_reference_never_reaches_the_workspace_or_the_instruction():
    for task_id in CELLS:
        for seed in range(30):
            task = skills.make_skill_task(task_id, seed)
            answer = skills.solve(task)
            assert task.output not in task.files
            visible = instruction(task, sorted(task.files), True) + "".join(task.files.values())
            assert answer.strip() not in visible
            for trap in task.private["family"].traps:
                assert trap not in instruction(task, sorted(task.files), True)


def test_a_pool_restricts_a_cell_to_named_families_and_rejects_others():
    for task_id in CELLS:
        names = skills.family_names(task_id)
        assert {skills.make_skill_task(task_id, s, pool=names[:1]).kind for s in range(20)} == {names[0]}
        with pytest.raises(ValueError):
            skills.make_skill_task(task_id, 1, pool=("no-such-family",))
    with pytest.raises(ValueError):
        skills.make_skill_task("qe_skill-office-d1", 1)


@pytest.mark.parametrize("task_id", CELLS)
def test_the_terminal_episode_grades_the_written_output_file(task_id):
    for seed in (3, 11):
        env = TerminalEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)
        task = skills.make_skill_task(task_id, seed)
        assert task.output in obs.instruction and str(MAX_STEPS) in obs.instruction
        listing = env.step(TerminalAction(action_type="exec", command="ls"))
        assert set(listing.output.split()) == set(task.files)
        env.step(TerminalAction(action_type="write_file", file_path=task.output, content=skills.solve(task)))
        obs = env.step(TerminalAction(action_type="evaluate"))
        assert obs.done and obs.reward == 1.0
        env = TerminalEnvironment()
        env.reset(seed=seed, task_id=task_id)
        assert env.step(TerminalAction(action_type="evaluate")).reward == 0.0
