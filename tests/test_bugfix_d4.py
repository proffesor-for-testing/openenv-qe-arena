"""qe_bugfix-d4: d3's faults plus a total edit budget, so the faults must be localised."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import solve_actions  # noqa: E402
from qe_arena import edit_budget, mutation, tasks  # noqa: E402
from qe_arena.environment import MAX_FREE_REFUSALS, MAX_STEPS, QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, make_task, parse_task_id  # noqa: E402

D4 = "qe_bugfix-d4"


def _env(seed):
    env = QEEnvironment()
    env.reset(seed=seed, task_id=D4)
    return env


def test_d4_is_a_bugfix_only_cell_appended_after_the_original_cells():
    assert tasks.TASK_IDS[:-1] == [f"{f}-d{lv}" for f in tasks.FAMILIES for lv in (1, 2, 3)]
    assert tasks.TASK_IDS[-1] == D4
    assert parse_task_id(D4) == ("qe_bugfix", 4)
    for bad in ("qe_mutant_kill-d4", "qe_bugfix-d5", "qe_bugfix-d0"):
        with pytest.raises(ValueError):
            parse_task_id(bad)
    assert tasks.format_task_id("qe_bugfix", 4) == D4
    with pytest.raises(ValueError):
        tasks.format_task_id("qe_mutant_kill", 4)
    assert tasks.FAULTS[4] == tasks.FAULTS[3] and tasks.MIN_SUBTLE[4] == tasks.MIN_SUBTLE[3]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_full_rewrite_is_refused_unapplied_and_free(seed):
    task = make_task(D4, seed)
    env = _env(seed)
    obs = env.step(QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=task.private["original"]))
    assert obs.error.startswith("edit refused") and env._files[MODULE] == task.private["buggy"]
    assert env.state.step_count == 0 and obs.remaining_steps == MAX_STEPS
    assert f"of the {task.private['edit_budget']} budget are left" in obs.error


def test_budget_arithmetic_is_exact():
    for length in (0, 1, 124, 125, 126, 1000, 1001):
        raw = math.ceil(edit_budget.BUDGET_SHARE * length)
        expected = -(-raw // edit_budget.BUDGET_QUANTUM) * edit_budget.BUDGET_QUANTUM
        assert edit_budget.budget_for("x" * length) == expected
        assert expected % edit_budget.BUDGET_QUANTUM == 0 and expected >= raw
    # Spending exactly the budget is allowed; one character more is refused; the reply shows what is left.
    task = make_task(D4, 3)
    budget, text = task.private["edit_budget"], task.private["buggy"]
    line = next(ln for ln in text.splitlines(keepends=True) if text.count(ln) == 1 and ln.strip())
    pad = budget - 2 * len(line)
    assert pad >= 0
    over = _env(3).step(QEAction(op="edit", path=MODULE, old=line, new=line + "#" * (pad + 1)))
    assert over.error.startswith("edit refused") and "it costs" in over.error
    assert f"{budget} of the {budget} budget" in over.error
    env = _env(3)
    exact = env.step(QEAction(op="edit", path=MODULE, old=line, new=line + "#" * pad))
    assert not exact.error and exact.content.endswith(f"edit budget left: 0 of {budget} characters")
    after = env.step(QEAction(op="edit", path=MODULE, old=line + "#" * pad, new=line))
    assert after.error.startswith("edit refused") and f"0 of the {budget} budget" in after.error


@pytest.mark.parametrize("seed", range(6))
def test_the_budget_reveals_nothing_about_the_faults(seed):
    """The shown number is a function of the shown file alone: equal for every fault set of that length."""
    task = make_task(D4, seed)
    budget = task.private["edit_budget"]
    assert budget == edit_budget.budget_for(task.files[MODULE])
    assert task.prompt.count(str(budget)) == 1 and f"Edit budget: {budget} characters" in task.prompt
    # Any other fault set on the same original with the same buggy length would show the same budget.
    other = "x" * len(task.files[MODULE])
    assert edit_budget.budget_for(other) == budget


def test_refused_edits_are_free_up_to_a_cap_so_episodes_stay_bounded():
    task = make_task(D4, 4)
    env = _env(4)
    rewrite = QEAction(op="edit", path=MODULE, old=task.private["buggy"], new=task.private["original"])
    for _ in range(MAX_FREE_REFUSALS):
        assert env.step(rewrite).error.startswith("edit refused")
    assert env.state.step_count == 0
    env.step(rewrite)
    assert env.state.step_count == 1  # past the cap a refusal uses a step


def test_lower_levels_carry_no_budget():
    for level in (1, 2, 3):
        task = make_task(f"qe_bugfix-d{level}", 0)
        assert task.private["edit_budget"] is None and "Edit budget" not in task.prompt


def test_d4_uses_its_own_random_stream():
    assert make_task(D4, 7).prompt != make_task("qe_bugfix-d3", 7).prompt
    assert random.Random(f"{D4}:7").random() != random.Random("qe_bugfix-d3:7").random()


# --------------------------------------------------------------------------- the localised-repair gate

GATE_BASE = (
    "import re\n\n\n"
    "def helper(x):\n    return x + 1\n\n\n"
    "def f(a, b):\n    \"\"\"Doc.\"\"\"\n    total = helper(a)\n    if total > b:\n        return b\n    return total\n"
)


@pytest.mark.parametrize("candidate, reason", [
    (GATE_BASE + "\n\ndef f(a, b):\n    return 0\n", "top-level"),
    (GATE_BASE + "\n\nf = lambda a, b: 0\n", "top-level"),
    (GATE_BASE.replace("import re\n", "import re\nimport math\n"), "top-level"),
    (GATE_BASE.replace("def f(a, b):", "def f(a, b, c=1):"), "definitions"),
    (GATE_BASE.replace("def f(a, b):", "@staticmethod\ndef f(a, b):"), "definitions"),
    (GATE_BASE.replace("    total = helper(a)\n", "    def g():\n        return 1\n    total = helper(a)\n"),
     "definitions"),
    (GATE_BASE.replace("    total = helper(a)\n", "    total = (lambda v: v)(helper(a))\n"), "lambda"),
    (GATE_BASE.replace("    return total\n", "    return total\n    total = 0\n"), "can never run"),
    (GATE_BASE.replace("    total = helper(a)\n", "    try:\n        return 0\n    finally:\n        pass\n    total = helper(a)\n"),
     "can never run"),
    (GATE_BASE.replace("    total = helper(a)\n", "    for _ in [0]:\n        return 0\n    total = helper(a)\n"), "can never run"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if a > b:\n        return 0\n    else:\n        return 1\n    total = helper(a)\n"),
     "can never run"),
    (GATE_BASE.replace("    total = helper(a)\n", "    while 1:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if {0}:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if ():\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if [a]:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if f'x':\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if a or 1:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if isinstance(a, object):\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if len(str(a)) >= 0:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if len(str(a)) > -1:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if not 0:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    if b is b:\n        return 0\n    total = helper(a)\n"),
     "constant condition"),
    (GATE_BASE.replace("    total = helper(a)\n", "    try:\n        return 0\n    except ValueError:\n        pass\n    total = helper(a)\n"),
     "can never run"),
    (GATE_BASE.replace("    if total > b:", "    if False:"), "constant condition"),
    (GATE_BASE.replace("    if total > b:", "    if not 0:"), "constant condition"),
    (GATE_BASE.replace("    if total > b:", "    if a == a:\n        return 1\n    if total > b:"), "constant condition"),
])
def test_gate_refuses_non_local_changes(candidate, reason):
    violation = edit_budget.structure_violation(GATE_BASE, candidate)
    assert violation is not None and reason in violation


def test_gate_accepts_a_local_repair_and_counts_tokens():
    fixed = GATE_BASE.replace("if total > b:", "if total >= b:")
    assert edit_budget.structure_violation(GATE_BASE, fixed) is None
    assert edit_budget.change_size(GATE_BASE, fixed) == 2  # one Gt out, one GtE in
    assert edit_budget.change_size(GATE_BASE, GATE_BASE) == 0
    assert edit_budget.change_allowance(GATE_BASE) == min(edit_budget.MAX_CHANGE, int(
        edit_budget.CHANGE_SHARE * edit_budget.code_tokens(GATE_BASE)))


def test_gate_refusal_costs_no_budget_and_grading_rechecks_the_gate():
    task = make_task(D4, 2)
    p = task.private
    env = _env(2)
    import ast  # noqa: PLC0415

    fn = next(n for n in ast.parse(p["buggy"]).body if isinstance(n, ast.FunctionDef) and n.name == p["entry"])
    first = fn.body[1] if ast.get_docstring(fn) is not None else fn.body[0]
    line = p["buggy"].splitlines(keepends=True)[first.lineno - 1]
    indent = line[:len(line) - len(line.lstrip())]
    new = indent + "return None\n" + line  # an early return: everything after it is dead code
    assert p["buggy"].count(line) == 1 and len(line) + len(new) <= p["edit_budget"]
    obs = env.step(QEAction(op="edit", path=MODULE, old=line, new=new))
    assert obs.error.startswith("edit refused") and "not a localised repair" in obs.error
    assert env._edit_spent == 0 and env.state.step_count == 0 and env._files[MODULE] == p["buggy"]
    shadow = p["buggy"] + f"\n\ndef {p['entry']}(*args):\n    return None\n"
    assert tasks.grade_bugfix(task, shadow) == 0.0  # whatever path produced it


def test_prompt_states_the_rules_and_the_refusal_cap():
    task = make_task(D4, 0)
    allowance = edit_budget.change_allowance(task.files[MODULE])
    assert f"change at most {allowance} syntax tokens" in task.prompt
    assert f"The first {MAX_FREE_REFUSALS} refusals do not use a step; later ones do." in task.prompt


def test_export_rows_keep_the_base_cells_byte_identical(tmp_path):
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "export_dataset.py"), str(tmp_path), "--seeds", "1"],
                   check=True, capture_output=True, timeout=900)
    rows = [json.loads(line) for line in (tmp_path / "tasks.jsonl").read_text().splitlines()]
    assert {r["task_id"] for r in rows} == set(tasks.TASK_IDS)
    for row in rows:
        assert ("edit_budget" in row) == (row["task_id"] == D4)


# --------------------------------------------------------------------------- false refusals of the true fix

# qe_bugfix-d4 seed 3000005 (baseline_verdict): var_swap made `if current_success > current_success:`
# (always false), and the oracle's first edit repaired the statement after it. The gate read the
# never-firing `if` as ending the block, and keyed the "dead" finding by the edited statement, so the
# true fix was refused as new dead code.
FALSE_TEST = (
    "def f(a, b):\n"
    "    if a > a:\n        return 1\n"
    "    if b and a >= b:\n        return 2\n"
    "    return 3\n"
)


def test_a_never_true_test_does_not_end_the_block():
    repaired = FALSE_TEST.replace("if b and a >= b:", "if a >= b:")
    assert edit_budget.structure_violation(FALSE_TEST, repaired) is None
    fully = repaired.replace("if a > a:", "if a > b:")
    assert edit_budget.structure_violation(FALSE_TEST, fully) is None


def test_repairing_code_after_a_fault_made_early_exit_is_not_new_dead_code():
    always = FALSE_TEST.replace("if a > a:", "if a <= a:")  # always true: what follows is dead in the buggy file
    assert edit_budget.structure_violation(always, always.replace("if b and a >= b:", "if a >= b:")) is None
    # A new early exit is still refused, before or after the existing one.
    inserted = always.replace("    return 3\n", "    return 4\n    return 3\n")
    assert edit_budget.structure_violation(always, inserted) is not None
    guard = always.replace("    if a <= a:", "    if {0}:\n        return 9\n    if a <= a:")
    assert edit_budget.structure_violation(always, guard) is not None


@pytest.mark.parametrize("seed", [3000005, 3000000, 3000001, 3000007, 3000011])
def test_oracle_passes_the_gate_on_a_fixed_sample_including_seed_3000005(seed):
    task = make_task(D4, seed)
    p = task.private
    current = p["buggy"]
    for edit in edit_budget.minimal_fix(p["buggy"], p["original"]):  # every intermediate file passes
        current = current.replace(edit["old"], edit["new"], 1)
        assert edit_budget.structure_violation(p["buggy"], current) is None, (seed, edit)
    env = QEEnvironment()
    obs = env.reset(seed=seed, task_id=D4)
    for action in solve_actions(task):
        obs = env.step(QEAction(**action))
        assert not obs.error, (seed, obs.error)
        if obs.done:
            break
    assert obs.done and obs.reward == 1.0
