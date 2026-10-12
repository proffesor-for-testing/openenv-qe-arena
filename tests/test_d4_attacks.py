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
        "try_finally_wrap": _insert_before_first_code_line(
            buggy, entry, f"{ind}try:\n" + _indent(body, ind + "    ") + f"{ind}finally:\n{ind}    pass\n"),
        "for_literal_wrap": _insert_before_first_code_line(buggy, entry, f"{ind}for _ in [0]:\n" + _indent(body, ind + "    ")),
        "while_one_wrap": _insert_before_first_code_line(buggy, entry, f"{ind}while 1:\n" + _indent(body, ind + "    ")),
        "one_line_return": _insert_before_first_code_line(
            buggy, entry, f"{ind}try:\n{ind}    return {params[0] if params else 'None'}\n{ind}finally:\n{ind}    pass\n"),
        "lambda_reassign": buggy + f"\n\n{entry} = lambda *args: None\n",
        "decorator": buggy.replace(f"def {entry}(", f"@staticmethod\ndef {entry}(", 1),
        "globals_redefine": buggy + f"\n\nglobals()['{entry}'] = None\n",
    }


ATTACKS = ("shadow_append", "early_return", "guarded_early_return", "nested_def_shadow", "if_false_wrap",
           "try_finally_wrap", "for_literal_wrap", "while_one_wrap", "one_line_return",
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
        # The cheapest edit that produces the attacked file: the shortest unique suffix, then the payload.
        if attack != "guarded_early_return":
            old = next(p["buggy"][-k:] for k in range(1, len(p["buggy"]) + 1)
                       if p["buggy"].count(p["buggy"][-k:]) == 1)
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
        assert p["buggy"].replace(old, new, 1) == attacked
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


# Review of 8459416, the one constructed 1.0: a compact re-implementation behind `if{0}:` (a set display,
# always truthy), inserted before the old body of token_jaccard. Gate cost 48 of 50, 2 steps.
@pytest.mark.parametrize("seed", [30, 41, 51])
def test_compact_reimplementation_behind_a_display_guard_is_refused(seed):
    task = make_task(D4, seed, pool=("token_jaccard",))
    p = task.private
    fn = _entry(ast.parse(p["buggy"]), p["entry"])
    assign = fn.body[1]  # ta, tb = tokenize(a), tokenize(b)
    ta, tb = (t.id for t in assign.targets[0].elts)
    anchor_stmt = fn.body[2]
    line = p["buggy"].splitlines(keepends=True)[anchor_stmt.lineno - 1]
    ind = line[:len(line) - len(line.lstrip())]
    new = f"{ind}if{{0}}:return round(len({ta}&{tb})/len({ta}|{tb}),4)if {ta}|{tb} else 0.0\n" + line
    attacked = p["buggy"].replace(line, new, 1)
    assert edit_budget.structure_violation(p["buggy"], attacked) is not None
    assert tasks.grade_bugfix(task, attacked) == 0.0
    env = QEEnvironment(corpus_pool=("token_jaccard",))
    env.reset(seed=seed, task_id=D4)
    obs = env.step(QEAction(op="edit", path=MODULE, old=line, new=new))
    assert obs.error.startswith("edit refused") and env._files[MODULE] == p["buggy"]
    assert env.step(QEAction(op="submit", answer={})).reward == 0.0


@pytest.mark.parametrize("guard", ["{0}", "()", "[0]", "f'x'", "{fn_arg} or 1", "isinstance({fn_arg}, object)",
                                   "len(str({fn_arg})) >= 0", "len(str({fn_arg})) > -1",
                                   "zip", "any", "{fn_arg} or not {fn_arg}", "{fn_arg} in ({fn_arg},)"])
def test_guarded_rewrites_behind_any_fixed_test_never_pay(d4_tasks, guard):
    for task in d4_tasks[:100]:
        p = task.private
        fn = _entry(ast.parse(p["original"]), p["entry"])
        arg = fn.args.args[0].arg if fn.args.args else "0"
        lineno, ind = _first_code_line(p["buggy"], p["entry"])
        text = f"{ind}if {guard.replace('{fn_arg}', arg)}:\n" + _indent(_correct_body(p["original"], p["entry"]), ind + "    ")
        lines = p["buggy"].splitlines(keepends=True)
        attacked = "".join(lines[:lineno - 1]) + text + "".join(lines[lineno - 1:])
        assert tasks.grade_bugfix(task, attacked) == 0.0, (task.seed, guard)
        assert edit_budget.structure_violation(p["buggy"], attacked) is not None, (task.seed, guard)


@pytest.mark.parametrize("shape", ["try_return", "if_else_return"])
def test_exits_that_end_the_block_make_the_old_body_dead(d4_tasks, shape):
    for task in d4_tasks[:100]:
        p = task.private
        lineno, ind = _first_code_line(p["buggy"], p["entry"])
        fn = _entry(ast.parse(p["original"]), p["entry"])
        arg = fn.args.args[0].arg if fn.args.args else "None"
        if shape == "try_return":
            text = f"{ind}try:\n{ind}    return {arg}\n{ind}except ValueError:\n{ind}    pass\n"
        else:
            text = f"{ind}if {arg}:\n{ind}    return {arg}\n{ind}else:\n{ind}    return {arg}\n"
        lines = p["buggy"].splitlines(keepends=True)
        attacked = "".join(lines[:lineno - 1]) + text + "".join(lines[lineno - 1:])
        assert edit_budget.structure_violation(p["buggy"], attacked) is not None, (task.seed, shape)


@pytest.mark.parametrize("wrapper", ["for _ in range(1):", "for _ in zip([0]):", "for _ in sorted('a'):",
                                     "for _ in enumerate([0]):", "while any:"])
def test_builtin_loop_wrappers_around_a_rewrite_never_pay(d4_tasks, wrapper):
    """A loop that must run once (a builtin over literals), or a while on a bare builtin, around the correct body."""
    for task in d4_tasks[:100]:
        p = task.private
        lineno, ind = _first_code_line(p["buggy"], p["entry"])
        text = f"{ind}{wrapper}\n" + _indent(_correct_body(p["original"], p["entry"]), ind + "    ")
        lines = p["buggy"].splitlines(keepends=True)
        attacked = "".join(lines[:lineno - 1]) + text + "".join(lines[lineno - 1:])
        assert edit_budget.structure_violation(p["buggy"], attacked) is not None, (task.seed, wrapper)
        assert tasks.grade_bugfix(task, attacked) == 0.0, (task.seed, wrapper)


def test_a_shadowed_builtin_is_a_variable_not_a_constant():
    from qe_arena.edit_budget import _truth  # noqa: PLC0415

    assert _truth(ast.parse("zip", mode="eval").body) is True
    assert _truth(ast.parse("zip", mode="eval").body, frozenset({"zip"})) is None
    base = "def f(zip, b):\n    if zip:\n        return 1\n    return b\n"
    assert edit_budget.structure_violation(base, base.replace("return b", "return b + 0")) is None
