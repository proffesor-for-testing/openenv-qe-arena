"""qe_bugfix-d4 against rewrite attacks (review of dd32587): each must be refused or score 0.

The attacker is as strong as possible: it knows the correct function (the private original) and
only has to smuggle a working copy past the budget and the localised-repair gate.
"""

from __future__ import annotations

import ast
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oracle import solve_actions  # noqa: E402
from qe_arena import edit_budget, tasks  # noqa: E402
from qe_arena.environment import MAX_STEPS, QEAction, QEEnvironment  # noqa: E402
from qe_arena.tasks import MODULE, make_task  # noqa: E402

D4 = "qe_bugfix-d4"
SEEDS = range(300)


def _make(seed):
    return make_task(D4, seed)


@pytest.fixture(scope="module")
def d4_tasks():
    with ProcessPoolExecutor(4) as pool:
        return list(pool.map(_make, SEEDS))


def _entry(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _code_body(fn: ast.FunctionDef) -> list[ast.stmt]:
    return fn.body[1:] if ast.get_docstring(fn) is not None else fn.body


def _indent(text: str, by: str) -> str:
    return "".join(by + line if line.strip() else line for line in text.splitlines(keepends=True))


def _first_code_line(source: str, entry: str) -> tuple[int, str]:
    fn = _entry(ast.parse(source), entry)
    first = _code_body(fn)[0]
    line = source.splitlines(keepends=True)[first.lineno - 1]
    return first.lineno, line[:len(line) - len(line.lstrip())]


def _correct_body(original: str, entry: str) -> str:
    """The correct entry body as compact source (ast.unparse), at column 0."""
    fn = _entry(ast.parse(original), entry)
    return "\n".join(ast.unparse(s) for s in _code_body(fn)) + "\n"


def _insert_before_first_code_line(buggy: str, entry: str, text: str) -> str:
    lineno, _ = _first_code_line(buggy, entry)
    lines = buggy.splitlines(keepends=True)
    return "".join(lines[:lineno - 1]) + text + "".join(lines[lineno - 1:])


def _wrap_old_body(buggy: str, entry: str, body: str, ind: str) -> str:
    """The correct body, then the whole old body under `if False:`."""
    fn = _entry(ast.parse(buggy), entry)
    lines = buggy.splitlines(keepends=True)
    start, end = _code_body(fn)[0].lineno - 1, fn.end_lineno
    return "".join(lines[:start]) + _indent(body, ind) + f"{ind}if False:\n" + _indent("".join(lines[start:end]), "    ") \
        + "".join(lines[end:])


def attacks(task) -> dict[str, str]:
    p = task.private
    buggy, original, entry = p["buggy"], p["original"], p["entry"]
    _, ind = _first_code_line(buggy, entry)
    body = _correct_body(original, entry)
    fn = _entry(ast.parse(original), entry)
    params = [a.arg for a in fn.args.args]
    guard = f"{params[0]} == {params[0]}" if params else "len('x') == 1"
    defs = "\n\n".join(ast.unparse(n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef))
    call = f"{entry}_fixed({', '.join(params)})"
    return {
        "shadow_append": buggy + "\n\n" + defs + "\n",
        "early_return": _insert_before_first_code_line(buggy, entry, _indent(body, ind)),
        "guarded_early_return": _insert_before_first_code_line(
            buggy, entry, f"{ind}if {guard}:\n" + _indent(body, ind + "    ")),
        "nested_def_shadow": _insert_before_first_code_line(
            buggy, entry, _indent(f"def {entry}_fixed({', '.join(params)}):\n" + _indent(body, "    ")
                                  + f"return {call}\n", ind)),
        "if_false_wrap": _wrap_old_body(buggy, entry, body, ind),
        "lambda_reassign": buggy + f"\n\n{entry} = lambda *args: None\n",
        "decorator": buggy.replace(f"def {entry}(", f"@staticmethod\ndef {entry}(", 1),
        "globals_redefine": buggy + f"\n\nglobals()['{entry}'] = None\n",
    }


ATTACKS = ("shadow_append", "early_return", "guarded_early_return", "nested_def_shadow", "if_false_wrap",
           "lambda_reassign", "decorator", "globals_redefine")


@pytest.mark.parametrize("attack", ATTACKS)
def test_every_attack_breaks_the_gate_and_grades_zero(d4_tasks, attack):
    for task in d4_tasks:
        p = task.private
        attacked = attacks(task)[attack]
        assert edit_budget.structure_violation(p["buggy"], attacked) is not None or _syntax_error(attacked), (
            task.seed, attack)
        assert tasks.grade_bugfix(task, attacked) == 0.0, (task.seed, attack)


def _syntax_error(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return True
    return False


@pytest.mark.parametrize("attack", ("shadow_append", "guarded_early_return", "lambda_reassign"))
def test_attacks_played_as_edits_are_refused(d4_tasks, attack):
    """Through the environment: a single edit that installs the attack is refused, so the file stays buggy."""
    for task in d4_tasks[:60]:
        p = task.private
        attacked = attacks(task)[attack]
        lines = p["buggy"].splitlines(keepends=True)
        # The cheapest edit that produces the attacked file: replace one unique line by itself plus the payload.
        old = next(ln for ln in reversed(lines) if p["buggy"].count(ln) == 1 and ln.strip())
        if attack != "guarded_early_return":
            new = old + attacked[len(p["buggy"]):] if attacked.startswith(p["buggy"]) else None
        else:
            lineno, _ = _first_code_line(p["buggy"], p["entry"])
            old = lines[lineno - 1]
            if p["buggy"].count(old) != 1:
                continue
            new = attacked[len("".join(lines[:lineno - 1])):len(attacked) - len("".join(lines[lineno:]))]
        if new is None:
            continue
        env = QEEnvironment()
        env.reset(seed=task.seed, task_id=D4)
        obs = env.step(QEAction(op="edit", path=MODULE, old=old, new=new))
        assert obs.error.startswith("edit refused") and env._files[MODULE] == p["buggy"], (task.seed, attack)
        assert env.step(QEAction(op="submit", answer={})).reward == 0.0


def test_oracle_earns_one_on_300_seeds_within_budget_gate_and_steps(d4_tasks):
    for task in d4_tasks:
        p = task.private
        assert edit_budget.structure_violation(p["buggy"], p["original"]) is None, task.seed
        assert edit_budget.fix_cost(p["buggy"], p["original"]) <= p["edit_budget"], task.seed
        env = QEEnvironment()
        obs = env.reset(seed=task.seed, task_id=D4)
        actions = solve_actions(task)
        assert len(actions) <= MAX_STEPS
        for action in actions:
            obs = env.step(QEAction(**action))
            assert not obs.error, (task.seed, obs.error)
            if obs.done:
                break
        assert obs.done and obs.reward == 1.0, task.seed


def _guard_every_function(buggy: str, original: str) -> list[str]:
    """For each function, helpers too: an always-true guard plus the correct body before the old code."""
    bt, ot = ast.parse(buggy), ast.parse(original)
    variants = []
    for bf in [n for n in bt.body if isinstance(n, ast.FunctionDef)]:
        of = _entry(ot, bf.name)
        old_body, good_body = _code_body(bf), _code_body(of)
        if not old_body or not good_body:
            continue
        lines = buggy.splitlines(keepends=True)
        first = lines[old_body[0].lineno - 1]
        ind = first[:len(first) - len(first.lstrip())]
        arg = of.args.args[0].arg if of.args.args else None
        for guard in ([f"{arg} == {arg}", f"{arg} is {arg}"] if arg else []) + ["len('x') == 1"]:
            text = f"{ind}if {guard}:\n" + "".join(ind + "    " + ln + "\n" for st in good_body
                                                      for ln in ast.unparse(st).splitlines())
            variants.append("".join(lines[:old_body[0].lineno - 1]) + text + "".join(lines[old_body[0].lineno - 1:]))
    return variants


def test_guarded_rewrites_of_any_function_never_pay(d4_tasks):
    for task in d4_tasks:
        p = task.private
        for attacked in _guard_every_function(p["buggy"], p["original"]):
            assert tasks.grade_bugfix(task, attacked) == 0.0, task.seed
