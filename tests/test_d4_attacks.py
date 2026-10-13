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
                                   "zip", "any", "{fn_arg} or not {fn_arg}", "{fn_arg} in ({fn_arg},)",
                                   "{entry}", "{helper}", "{module}", "{fn_arg} or {helper}", "not not {module}",
                                   "{module}.__name__"])
def test_guarded_rewrites_behind_any_fixed_test_never_pay(d4_tasks, guard):
    played = 0
    needs_object = "{helper}" in guard or "{module}" in guard  # only some files have a helper or an import
    for task in d4_tasks if needs_object else d4_tasks[:100]:
        p = task.private
        fn = _entry(ast.parse(p["original"]), p["entry"])
        names = {"fn_arg": fn.args.args[0].arg if fn.args.args else "0", "entry": p["entry"]}
        names.update(_object_names(p["buggy"], p["entry"]))
        test = guard
        for key, value in names.items():
            test = test.replace("{" + key + "}", value)
        if "{helper}" in test or "{module}" in test:
            continue  # no helper function or import in this file
        played += 1
        lineno, ind = _first_code_line(p["buggy"], p["entry"])
        text = f"{ind}if {test}:\n" + _indent(_correct_body(p["original"], p["entry"]), ind + "    ")
        lines = p["buggy"].splitlines(keepends=True)
        attacked = "".join(lines[:lineno - 1]) + text + "".join(lines[lineno - 1:])
        assert tasks.grade_bugfix(task, attacked) == 0.0, (task.seed, guard)
        assert edit_budget.structure_violation(p["buggy"], attacked) is not None, (task.seed, guard)
    assert played >= 10, (guard, played)


def _object_names(source: str, entry: str) -> dict:
    """A helper function and an imported module of the file, when it has them (bound only by def/import)."""
    tree = ast.parse(source)
    names = {}
    helpers = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != entry]
    modules = [(a.asname or a.name).split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names]
    bound = edit_budget._bound_names(tree)
    if helpers and edit_budget.OBJECT + helpers[0] in bound:
        names["helper"] = helpers[0]
    if modules and edit_budget.OBJECT + modules[0] in bound:
        names["module"] = modules[0]
    return names


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


def _oracle_reward(seed):
    task = make_task(D4, seed)
    env = QEEnvironment()
    obs = env.reset(seed=seed, task_id=D4)
    errors = []
    for action in solve_actions(task):
        obs = env.step(QEAction(**action))
        if obs.error:
            errors.append(obs.error)
        if obs.done:
            break
    return seed, obs.reward, errors


@pytest.mark.parametrize("chunk", range(10))
def test_oracle_earns_one_on_1000_natural_seeds(chunk):
    """No false refusal of the true minimal fix on 1000 seeds outside every other test's range.

    Ten chunks of 100 seeds, run serially in each case, so xdist and the CI shards spread them; one
    1000-seed case with its own process pool was the critical path of a slow shard (99 of 120 min)."""
    start = 3_000_000 + 100 * chunk
    results = [_oracle_reward(seed) for seed in range(start, start + 100)]
    failed = [(seed, reward, errors[:1]) for seed, reward, errors in results if reward != 1.0 or errors]
    assert not failed, failed[:10]


# Review of 20eebdf: a name bound only by def or import (`token_jaccard`, `tokenize`, `re`) is a function or
# module object, so `if token_jaccard:` is as constant as `if{0}:`. The gate read it as a variable test,
# and natural seed 3000076 paid 1.0 in two edits at change size 50 of 50.
SEED_3000076_COMPACT = "    if token_jaccard:return round(len(seen&group)/len(seen|group),4)if seen|group else 0.0\n"


def test_seed_3000076_compact_return_behind_the_entry_name_is_refused():
    task = make_task(D4, 3_000_076)
    p = task.private
    assert p["corpus"] == "token_jaccard" and "text.upper()" in p["buggy"]
    anchor = "    if seen and not seen and not group:\n"
    assert anchor in p["buggy"]
    env = QEEnvironment()
    env.reset(seed=3_000_076, task_id=D4)
    assert not env.step(QEAction(op="edit", path=MODULE, old="upp", new="low")).error
    before = env._files[MODULE]
    short = "    if seen and"  # a short anchor, so the edit fits the character budget
    assert before.count(short) == 1
    obs = env.step(QEAction(op="edit", path=MODULE, old=short, new=SEED_3000076_COMPACT + short))
    assert obs.error.startswith("edit refused") and "constant condition" in obs.error
    assert env._files[MODULE] == before
    assert env.step(QEAction(op="submit", answer={})).reward == 0.0
    attacked = before.replace(anchor, SEED_3000076_COMPACT + anchor, 1)
    assert tasks.grade_bugfix(task, attacked) == 0.0


@pytest.mark.parametrize("guard", ["token_jaccard", "tokenize", "re", "seen or tokenize", "re.split"])
def test_def_and_import_names_are_constant_even_if_the_candidate_rebinds_them(guard):
    task = make_task(D4, 3_000_076)
    buggy = task.private["buggy"]
    anchor = "    if seen and not seen and not group:\n"
    guarded = buggy.replace(anchor, f"    if {guard}:\n        return 0.0\n" + anchor, 1)
    assert "constant condition" in edit_budget.structure_violation(buggy, guarded)
    # Rebinding the name in another function does not make it a variable again.
    rebound = guarded.replace("def tokenize(text):\n", "def tokenize(text):\n    token_jaccard = tokenize = re = 0\n", 1)
    assert "constant condition" in edit_budget.structure_violation(buggy, rebound)


def test_a_def_name_that_is_also_assigned_stays_a_variable():
    source = (
        "def helper(x):\n    return x\n\n\n"
        "def f(a):\n    global helper\n    if helper:\n        return 1\n    helper = None\n    return 2\n"
    )
    assert edit_budget.OBJECT + "helper" not in edit_budget._bound_names(ast.parse(source))
    assert not edit_budget._dead_code(ast.parse(source))
    for binder in ("helper = 0", "for helper in a: pass", "with a as helper: pass", "del helper",
                   "(helper := 1)", "[helper for helper in a]"):
        src = f"def helper(x):\n    return x\n\n\ndef f(a):\n    {binder}\n    if helper:\n        return 1\n    return 2\n"
        assert edit_budget.OBJECT + "helper" not in edit_budget._bound_names(ast.parse(src)), binder
        assert not any(k.startswith("const:") for k, _, _ in edit_budget._dead_code(ast.parse(src))), binder


# Review of 3c53ade: attack plays at the merged head found guards that still passed the constant check,
# so only the 50-token cap stopped them (2 tokens of margin): `type(v)`, `repr(v)` and `(_q := re)` at
# 52-54 tokens, and `__name__`, `__debug__` and `__import__` at 48 of 50, which earned 0 only because
# the sandbox rejects dunder names. The rest were found by fuzzing guard shapes against the gate.
REVIEW_GUARDS = ("type(seen)", "repr(seen)", "(_q := re)", "__name__", "__debug__", "__import__")
FUZZED_GUARDS = ("ascii(seen)", "id(seen)", "object()", "zip()", "globals()", "locals()", "ValueError(seen)",
                 "not print(seen)", "__builtins__", "__spec__", "seen.__class__", "seen.__eq__", "seen is not []",
                 "seen not in ()", "(t for t in seen)", "getattr(seen, '__class__')", "hasattr(seen, '__eq__')",
                 "isinstance(seen, (set, object))", "issubclass(type(seen), object)", "1 if seen else 2",
                 "vars(re)", "bool(type(seen))", "(_q := type(seen))", "not exec('')")


@pytest.mark.parametrize("guard", REVIEW_GUARDS + FUZZED_GUARDS)
def test_seed_3000076_compact_return_behind_an_always_true_call_or_dunder_is_refused(guard):
    task = make_task(D4, 3_000_076)
    p = task.private
    compact = SEED_3000076_COMPACT.replace("if token_jaccard:", f"if {guard}:")
    anchor = "    if seen and not seen and not group:\n"
    attacked = p["buggy"].replace("upp", "low", 1).replace(anchor, compact + anchor, 1)
    assert "constant condition" in edit_budget.structure_violation(p["buggy"], attacked), guard
    assert tasks.grade_bugfix(task, attacked) == 0.0
    env = QEEnvironment()
    env.reset(seed=3_000_076, task_id=D4)
    assert not env.step(QEAction(op="edit", path=MODULE, old="upp", new="low")).error
    before = env._files[MODULE]
    short = "    if seen and"
    obs = env.step(QEAction(op="edit", path=MODULE, old=short, new=compact + short))
    assert obs.error.startswith("edit refused") and "constant condition" in obs.error, (guard, obs.error)
    assert env._files[MODULE] == before
    assert env.step(QEAction(op="submit", answer={})).reward == 0.0


@pytest.fixture(scope="module")
def jaccard_tasks():
    return {seed: make_task(D4, seed, pool=("token_jaccard",)) for seed in (30, 41, 51)}


@pytest.mark.parametrize("guard", ("type({ta})", "repr({ta})", "(_q := re)", "__name__", "__debug__", "__import__",
                                   "zip()", "{ta} is not []"))
@pytest.mark.parametrize("seed", [30, 41, 51])
def test_compact_reimplementation_behind_a_call_walrus_or_dunder_guard_is_refused(jaccard_tasks, seed, guard):
    task = jaccard_tasks[seed]
    p = task.private
    fn = _entry(ast.parse(p["buggy"]), p["entry"])
    ta, tb = (t.id for t in fn.body[1].targets[0].elts)
    line = p["buggy"].splitlines(keepends=True)[fn.body[2].lineno - 1]
    ind = line[:len(line) - len(line.lstrip())]
    test = guard.format(ta=ta)
    new = f"{ind}if {test}:return round(len({ta}&{tb})/len({ta}|{tb}),4)if {ta}|{tb} else 0.0\n" + line
    attacked = p["buggy"].replace(line, new, 1)
    assert "constant condition" in edit_budget.structure_violation(p["buggy"], attacked), (seed, guard)
    assert tasks.grade_bugfix(task, attacked) == 0.0


@pytest.mark.parametrize("guard", ["type(seen) is int", "repr(seen) == 'x'", "isinstance(seen, int)", "(_q := seen)",
                                   "hasattr(seen, 'x')", "bool(seen)", "[*seen]", "1 if seen else 0",
                                   "type(seen) is type(group)", "id(seen) == id(group)", "getattr(seen, 'x', 0)"])
def test_a_test_that_reads_the_input_through_a_builtin_stays_a_variable_test(guard):
    """Negative control: only the result's truth is fixed; comparing it, or a truth that follows the input, is not."""
    task = make_task(D4, 3_000_076)
    buggy = task.private["buggy"]
    anchor = "    if seen and not seen and not group:\n"
    guarded = buggy.replace(anchor, f"    if {guard}:\n        return 0.0\n" + anchor, 1)
    assert edit_budget.structure_violation(buggy, guarded) is None, guard


def test_a_builtin_the_file_shadows_or_whose_special_method_it_defines_is_not_trusted():
    from qe_arena.edit_budget import _bound_names, _truth  # noqa: PLC0415

    def truth(expr, source):
        return _truth(ast.parse(expr, mode="eval").body, _bound_names(ast.parse(source)))

    plain = "def f(v):\n    return v\n"
    assert truth("type(v)", plain) is True and truth("repr(v)", plain) is True and truth("print(v)", plain) is False
    assert truth("type(v)", "def f(type, v):\n    return v\n") is None  # a parameter named type
    assert truth("repr(v)", plain + "\n\ndef __repr__(self):\n    return ''\n") is None
    assert truth("type(v)", plain + "\n\ndef __bool__(self):\n    return False\n") is None
    assert truth("__name__", plain) == edit_budget.FIXED
    assert truth("__name__", "__name__ = 'x'\n" + plain) is None  # assigned: a variable like any other
    assert truth("eval('v')", plain) is None  # reads a variable through a string
    assert truth("vars()", plain) == edit_budget.FIXED and truth("vars(v)", plain) is None
