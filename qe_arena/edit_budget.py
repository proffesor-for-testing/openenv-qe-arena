"""qe_bugfix-d4: an edit budget plus a structural gate, so only a localised repair can pass.

Budget. BUDGET_SHARE of the shown file's length, rounded up to BUDGET_QUANTUM characters, counted as
len(old) + len(new) over all edits. It depends on the shown file only, so the number says nothing
about the faults. A budget computed from the faults (the 1.5x line-diff prototype) would have told the
model how large the faulty code is. A task is served only if the character-level minimal fix fits.

Gate. A budget alone does not stop a rewrite: a minified copy of the function appended after the
old one, or placed before it behind an early return, fits in a few hundred characters (review of
dd32587). Every edit, and the graded file, must keep the buggy file's structure:
- the same top-level statements in the same order, and the same function definitions (names,
  arguments, decorators), so there is no shadow definition and no new top-level code;
- no lambda beyond those already in the file;
- every statement stays reachable (a static check that proves exits through returns, raises,
  if/else, try/finally, with, and loops that must run their body), and no branch or loop tests a
  constant or a tautology such as `p == p`, so no wrapper can bypass the old body;
- at most change_allowance(buggy) syntax tokens changed against the buggy file (a sequence diff of
  the AST in preorder): CHANGE_SHARE of the file's code tokens, capped at MAX_CHANGE. It depends on
  the shown file only, so the gate reveals nothing about the faults. A compact re-implementation of a
  function needs more tokens than that, even behind an always-true guard.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import math
from collections import Counter

BUDGET_SHARE = 0.25
BUDGET_QUANTUM = 25  # characters
MERGE_GAP = 16  # character hunks closer than this become one edit
CHANGE_SHARE = 0.35  # of the shown file's code tokens (docstrings excluded)
MAX_CHANGE = 50  # AST tokens


# --------------------------------------------------------------------------- the minimal fix

def _char_hunks(buggy: str, original: str) -> list[list[int]]:
    ops = [op for op in difflib.SequenceMatcher(None, buggy, original, autojunk=False).get_opcodes()
           if op[0] != "equal"]
    merged: list[list[int]] = []
    for _, i1, i2, j1, j2 in ops:
        if merged and i1 - merged[-1][1] < MERGE_GAP:
            merged[-1][1], merged[-1][3] = i2, j2
        else:
            merged.append([i1, i2, j1, j2])
    return merged


def minimal_fix(buggy: str, original: str) -> list[dict]:
    """Character-level edits, applied in order, that turn ``buggy`` into ``original``.

    Changes closer than MERGE_GAP characters form one edit. Edits run last to first, and each is
    widened one character at a time on both sides until its ``old`` occurs exactly once in the file
    as it stands when that edit is applied.
    """
    current = buggy
    edits = []
    for i1, i2, j1, j2 in reversed(_char_hunks(buggy, original)):
        lo, hi = i1, i2
        while not (current[lo:hi] and current.count(current[lo:hi]) == 1):
            if lo == 0 and hi == len(current):
                break
            lo, hi = max(0, lo - 1), min(len(current), hi + 1)
        new = current[lo:i1] + original[j1:j2] + current[i2:hi]
        edits.append({"old": current[lo:hi], "new": new})
        current = current[:lo] + new + current[hi:]
    if current != original:  # pragma: no cover - defensive invariant
        raise AssertionError("minimal_fix does not restore the original")
    return edits


def edit_cost(edit: dict) -> int:
    return len(edit["old"]) + len(edit["new"])


def fix_cost(buggy: str, original: str) -> int:
    return sum(edit_cost(e) for e in minimal_fix(buggy, original))


def budget_for(shown: str) -> int:
    """The budget for a file: a function of the shown text only, never of the faults."""
    raw = math.ceil(BUDGET_SHARE * len(shown))
    return -(-raw // BUDGET_QUANTUM) * BUDGET_QUANTUM


# --------------------------------------------------------------------------- the structural gate

def _tokens(node: ast.AST) -> list[str]:
    """Preorder AST tokens: node types plus identifiers, constants and operators."""
    out = [type(node).__name__]
    for name in ("id", "arg", "attr", "name", "module"):
        value = getattr(node, name, None)
        if isinstance(value, str):
            out.append(f"{name}={value}")
    if isinstance(node, ast.Constant):
        out.append(f"const={node.value!r}")
    for child in ast.iter_child_nodes(node):
        out.extend(_tokens(child))
    return out


def change_size(buggy: str, candidate: str) -> int:
    """Tokens inserted plus tokens deleted between the two files' preorder AST token sequences."""
    a, b = _tokens(ast.parse(buggy)), _tokens(ast.parse(candidate))
    matched = sum(block.size for block in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return len(a) + len(b) - 2 * matched


def code_tokens(source: str) -> int:
    """Preorder AST tokens of ``source`` with docstrings left out."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return len(_tokens(tree))


def change_allowance(shown: str) -> int:
    return min(MAX_CHANGE, int(CHANGE_SHARE * code_tokens(shown)))


def _top_level(tree: ast.Module) -> list[tuple]:
    return [(type(n).__name__, getattr(n, "name", None), ast.dump(n) if isinstance(n, (ast.Import, ast.ImportFrom)) else None)
            for n in tree.body]


def _definitions(tree: ast.AST) -> list[tuple]:
    return [(n.name, ast.dump(n.args), [ast.dump(d) for d in n.decorator_list])
            for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


FIXED = "fixed"  # a test whose value cannot depend on the input, but which we do not evaluate


def _truth(test: ast.AST, bound: frozenset = frozenset()) -> bool | str | None:
    """The truth value of ``test`` when it cannot depend on the input, FIXED when it cannot depend on
    the input but its value is not computed here, else None.

    Tautologies are open-ended: this catches the shapes a rewrite needs to hide behind, and the
    token allowance bounds the rest. Caught: literals and displays (`{0}`, `()`, f-strings),
    anything with no variable in it (builtins and module dunders such as `__name__` are not
    variables), a bare builtin such as `zip` (when no name in the file shadows it), a walrus by its
    value, `x OP x`, `x or not x`, `x in (x,)`, `x in ()`, `x is []`, `x or <truthy>` /
    `x and <falsy>`, `isinstance(_, object)`, `len(...) >= 0`, a call to a builtin whose result is
    truthy whatever the argument (`type(x)`, `repr(x)`, `ValueError(x)`, see _TRUTHY_CALLS) or always
    None (`print(x)`), a special method such as `x.__eq__`, a generator expression, and
    `a if x else b` with a and b of the same truth. ``bound`` is from _bound_names: every name the
    file binds as a variable, plus OBJECT + name for a name bound only by def, class or import.
    """
    if isinstance(test, ast.NamedExpr):
        return _truth(test.value, bound)  # (q := re) is as true as re
    if isinstance(test, ast.Name) and (OBJECT + test.id in bound or (test.id in _BUILTINS and test.id not in bound)):
        return True  # a function, class, module or builtin object is always truthy
    if isinstance(test, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
        items = test.keys if isinstance(test, ast.Dict) else test.elts
        if not items or any(i is not None and not isinstance(i, ast.Starred) for i in items):
            return bool(items)  # `[*x]` and `{**x}` are as empty as x: decided below
    if isinstance(test, ast.JoinedStr):
        if any(isinstance(v, ast.Constant) and v.value for v in test.values):
            return True
        return None if any(isinstance(v, ast.FormattedValue) for v in test.values) else False
    if not _reads_input(test, bound):
        try:
            return bool(ast.literal_eval(test))
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return FIXED  # no variable in it: fixed, but not evaluated here (never run model code)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = _truth(test.operand, bound)
        return inner if inner is None or inner == FIXED else not inner
    if isinstance(test, ast.BoolOp):
        values = [_truth(v, bound) for v in test.values]
        decisive = True if isinstance(test.op, ast.Or) else False
        if decisive in values:
            return decisive
        dumps = [ast.dump(v) for v in test.values]
        negated = {ast.dump(v.operand) for v in test.values if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not)}
        if negated & set(dumps):
            return decisive  # x or not x is always true; x and not x is always false
        if decisive is True and FIXED in values:
            return FIXED  # x or <fixed>: still refused as constant, as when <fixed> counted as true
        if None in values:
            return None
        return FIXED if FIXED in values else not decisive
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if isinstance(op, (ast.Is, ast.IsNot)) and (isinstance(left, _FRESH) or isinstance(right, _FRESH)):
            return isinstance(op, ast.IsNot)  # a new list, dict, set or generator is nothing else
        if ast.dump(left) == ast.dump(right):
            if isinstance(op, (ast.In, ast.NotIn)):
                return None  # x in x is not fixed for lists
            return isinstance(op, (ast.Eq, ast.Is, ast.LtE, ast.GtE))
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            items = right.keys if isinstance(right, ast.Dict) else right.elts
            if not items:
                return isinstance(op, ast.NotIn)  # nothing is in an empty display
            if any(ast.dump(e) == ast.dump(left) for e in items):
                return isinstance(op, ast.In)  # x in (x,) is always true
        if _is_len(left) and _int_value(right) is not None:
            limit = _int_value(right)
            if (isinstance(op, ast.GtE) and limit <= 0) or (isinstance(op, ast.Gt) and limit < 0):
                return True
        if _is_len(right) and _int_value(left) is not None:
            limit = _int_value(left)
            if (isinstance(op, ast.LtE) and limit <= 0) or (isinstance(op, ast.Lt) and limit < 0):
                return True
    if (isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id in ("isinstance", "issubclass")
            and len(test.args) == 2 and _is_object(test.args[1])):
        return True  # everything is an object
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and _builtin(test.func.id, bound):
        name, args = test.func.id, test.args
        if name in _NONE_CALLS:
            return False
        if name in _TRUTHY_CALLS and not (name in _SCOPE_READERS and not args) and _trusted(name, bound):
            return True
        if name in _SCOPE_READERS and not args and not test.keywords:
            return FIXED  # the names bound in this scope: never what a repair tests
        if name == "bool" and len(args) == 1:
            return _truth(args[0], bound)
        if name in ("getattr", "hasattr") and len(args) == 2 and isinstance(args[1], ast.Constant) \
                and args[1].value in (_SPECIAL_METHODS if name == "getattr" else _OBJECT_ATTRIBUTES) \
                and _trusted(args[1].value, bound):
            return True  # getattr(x, "__class__") is x.__class__; every object has what object has
    if isinstance(test, ast.Attribute) and test.attr in _SPECIAL_METHODS and _trusted(test.attr, bound):
        return True  # x.__class__ is a class and x.__eq__ a bound method, whatever x is
    if isinstance(test, ast.GeneratorExp):
        return True  # a generator object is always truthy
    if isinstance(test, ast.IfExp):
        body, orelse = _truth(test.body, bound), _truth(test.orelse, bound)
        if isinstance(body, bool) and body is orelse:
            return body
    return None


_BUILTINS = frozenset(n for n in dir(builtins) if not n.startswith("_"))
_FRAME_READERS = frozenset({"eval", "exec"})  # read any variable named in a string
_SCOPE_READERS = frozenset({"globals", "locals", "vars", "dir"})  # read the variables in scope when called bare
_OBJECT_ATTRIBUTES = frozenset(dir(object))
_FRESH = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# Builtin calls whose result is truthy for every argument they accept: a builtin class whose instances
# define neither __bool__ nor __len__ (type, object, zip, ValueError, ...), a positive id, a repr or
# ascii (never empty), an iterator, a non-empty dir(x), and the module globals (never empty).
_TRUTHY_CALLS = frozenset(name for name, value in vars(builtins).items() if isinstance(value, type)
                          and not hasattr(value, "__bool__") and not hasattr(value, "__len__")) \
    | {"id", "repr", "ascii", "iter", "dir", "globals"}
_NONE_CALLS = frozenset({"print", "setattr", "delattr", "exec"})  # always return None
# Special methods a class in the file could define so that the builtin above returns a falsy value.
_DELEGATES = {"type": ("__bool__", "__len__"), "__class__": ("__bool__", "__len__"),
              "repr": ("__repr__",), "ascii": ("__repr__",), "dir": ("__dir__",),
              "iter": ("__iter__", "__bool__", "__len__"), "reversed": ("__reversed__", "__bool__", "__len__")}


def _special_methods() -> frozenset:
    """Special methods callable on every builtin type that has them: looked up on any object, each is a
    bound method or (__class__) a class, so always truthy. __hash__ is None on unhashable types, and
    data attributes such as __doc__ or __name__ vary, so neither is here."""
    kinds = [v for v in vars(builtins).values() if isinstance(v, type)] + [type(None), type(len), type(_special_methods)]
    names = {n for kind in kinds for n in dir(kind) if n.startswith("__") and n.endswith("__")}
    owners = {n: [kind for kind in kinds if hasattr(kind, n)] for n in names}
    return frozenset(n for n in names if owners[n] and all(callable(getattr(kind, n)) for kind in owners[n]))


_SPECIAL_METHODS = _special_methods()


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _fixed_name(name: str, bound: frozenset) -> bool:
    """A name that cannot carry the input: a def, class or import object, a builtin no name in the file
    shadows (except those that read variables by name), or a module dunder such as __name__."""
    if name in bound:
        return False
    if OBJECT + name in bound:
        return True
    return (name in vars(builtins) and name not in _FRAME_READERS) or _is_dunder(name)


def _reads_input(test: ast.AST, bound: frozenset) -> bool:
    """Whether ``test`` can see the input: a name that is not fixed, or a bare `locals()` or `vars()`."""
    for node in ast.walk(test):
        if isinstance(node, ast.Name) and not _fixed_name(node.id, bound):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SCOPE_READERS \
                and not node.args and not node.keywords and _builtin(node.func.id, bound):
            return True
    return False


def _is_object(node: ast.AST) -> bool:
    """`object`, or a tuple display that names it (isinstance(x, (int, object)))."""
    if isinstance(node, ast.Tuple):
        return any(_is_object(e) for e in node.elts)
    return isinstance(node, ast.Name) and node.id == "object"


def _builtin(name: str, bound: frozenset) -> bool:
    return name in vars(builtins) and name not in bound and OBJECT + name not in bound


def _trusted(name: str, bound: frozenset) -> bool:
    """No function in the file defines a special method that could make ``name``'s result falsy."""
    return not any(m in bound or OBJECT + m in bound for m in _DELEGATES.get(name, ()))


def _is_len(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len"


def _int_value(node: ast.AST) -> int | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    return value if type(value) is int else None


def _constant_test(test: ast.AST, bound: frozenset = frozenset()) -> bool:
    """A test whose value does not depend on the input (see _truth)."""
    return _truth(test, bound) is not None


OBJECT = "object:"  # prefix in _bound_names for a name that only def, class or import binds


def _bound_names(tree: ast.AST) -> frozenset:
    """Names the file binds as variables, plus OBJECT + name for each name bound only by def, class or
    import: always a truthy object, so `if tokenize:` or `if re:` is as constant as `if {0}:`. A name
    that is also assigned, a parameter, a loop or with/except target, deleted, or declared global or
    nonlocal anywhere stays a variable."""
    variables, objects = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            variables.add(node.id)  # assignment, augmented, walrus, for/with/comprehension target, del
        elif isinstance(node, ast.arg):
            variables.add(node.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            variables.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            variables.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            variables.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            variables.add(node.rest)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            objects.add(node.name)
        elif isinstance(node, ast.alias):
            objects.add((node.asname or node.name).split(".")[0])
    return frozenset(variables | {OBJECT + name for name in objects - variables})


_RUNS = ("range", "zip", "sorted", "reversed", "list", "tuple", "enumerate", "str", "set", "map", "filter")


def _must_iterate(node: ast.AST, bound: frozenset) -> bool:
    """A for-loop iterable that is certainly non-empty: a non-empty literal, or a builtin over literals."""
    if _nonempty_literal(node):
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _RUNS
            and node.func.id not in bound and OBJECT + node.func.id not in bound and not node.keywords):
        try:
            args = [ast.literal_eval(a) for a in node.args]
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return False
        try:
            if node.func.id == "range":
                return all(type(a) is int and abs(a) < 10**9 for a in args) and len(range(*args)) > 0
            if node.func.id in ("map", "filter"):
                return False  # needs a function; never a plain wrapper
            if any(not isinstance(a, (str, bytes, list, tuple, set, dict)) for a in args) or len(str(args)) > 1000:
                return False
            return len(list(getattr(builtins, node.func.id)(*args))) > 0
        except Exception:  # noqa: BLE001 - a call that fails on literals is not a wrapper we can prove
            return False
    return False


def _loop_body_jumps_out(body: list[ast.stmt]) -> bool:
    """A break that leaves this loop (not one inside a nested loop or function)."""
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Break):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _nonempty_literal(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return bool(node.value)
    return False


def _ends(block: list[ast.stmt], jumps: bool, bound: frozenset = frozenset()) -> bool:
    """True if control can never fall off the end of ``block``.

    ``jumps`` says whether break/continue end it (they do inside a loop body). The analysis is
    sound for exits it proves: a return, a raise, an if with every branch ending, a try whose
    body and handlers end or whose finally ends, a with whose body ends, and a loop that must run
    its body (a constant-true while, a for over a non-empty literal) with a body that ends and
    no break. A branch on a test that is not constant may go either way.
    """
    return any(_stmt_ends(stmt, jumps, bound) for stmt in block)


def _stmt_ends(stmt: ast.stmt, jumps: bool, bound: frozenset = frozenset()) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, (ast.Break, ast.Continue)):
        return jumps
    if isinstance(stmt, ast.If):
        value = _truth(stmt.test, bound)
        if value is True:
            return _ends(stmt.body, jumps, bound)
        if value is False:
            return _ends(stmt.orelse, jumps, bound)  # the body never runs: `if x > x:` cannot end the block
        if value == FIXED:
            return _ends(stmt.body, jumps, bound) or _ends(stmt.orelse, jumps, bound)  # either may be live
        return bool(stmt.orelse) and _ends(stmt.body, jumps, bound) and _ends(stmt.orelse, jumps, bound)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _ends(stmt.body, jumps, bound)
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        # Conservative: a try whose body or finally ends is treated as ending, even if a handler could
        # fall through (`try: return X / except: pass`). The corpus never needs that shape, and a
        # rewrite hides behind it; the d4 prompt says so.
        return _ends(stmt.body, jumps, bound) or _ends(stmt.finalbody, jumps, bound)
    if isinstance(stmt, ast.While):
        value = _truth(stmt.test, bound)
        return value in (True, FIXED) and not _loop_body_jumps_out(stmt.body)  # a false while never runs
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return (_must_iterate(stmt.iter, bound) and _ends(stmt.body, False, bound)
                and not _loop_body_jumps_out(stmt.body))
    return False


def _dead_code(tree: ast.AST, objects: frozenset = frozenset()) -> list[tuple[str, str, str]]:
    """Constant or tautological tests (a wrapper such as `if True:`), then unreachable statements.

    ``objects`` are names the buggy file binds only by def, class or import. They stay objects here
    even if the candidate also assigns them, so adding `tokenize = 0` somewhere cannot turn
    `if tokenize:` back into a variable test.

    Each finding is (key, location, message). The key identifies the code itself, not its line,
    so findings the buggy file already has (a fault can create `x > x`) can be told apart from
    new ones.
    """
    found = []
    bound = (_bound_names(tree) - objects) | {OBJECT + name for name in objects}
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp)) and _constant_test(node.test, bound):
            found.append((f"const:{ast.dump(node.test)}", str(node.lineno), f"a constant condition at line {node.lineno}"))
    for node in ast.walk(tree):
        blocks = [getattr(node, field, None) for field in ("body", "orelse", "finalbody")]
        blocks += [h.body for h in getattr(node, "handlers", None) or []]
        for block in blocks:
            if not isinstance(block, list) or not all(isinstance(s, ast.stmt) for s in block):
                continue
            for stmt, nxt in zip(block, block[1:]):
                if _stmt_ends(stmt, True, bound):
                    # Keyed by the statement that ends the block, not by what follows it: the buggy file
                    # may already end a block early (a fault can make `x <= x`), and repairing the code
                    # after that statement must not read as new dead code.
                    found.append((f"dead-after:{ast.dump(stmt)}", str(nxt.lineno),
                                  f"line {nxt.lineno} can never run (after line {stmt.lineno})"))
    return found


def structure_violation(buggy: str, candidate: str) -> str | None:
    """Why ``candidate`` is not a localised repair of ``buggy``, or None if it is."""
    try:
        tree = ast.parse(candidate)
    except SyntaxError:
        return None  # a syntax error is the candidate's own failure; grading scores it 0
    base = ast.parse(buggy)
    if _top_level(tree) != _top_level(base):
        return "top-level code must stay as it is (no new or redefined functions, imports or statements)"
    if _definitions(tree) != _definitions(base):
        return "function definitions, arguments and decorators must stay as they are"
    if sum(isinstance(n, ast.Lambda) for n in ast.walk(tree)) > sum(isinstance(n, ast.Lambda) for n in ast.walk(base)):
        return "no new lambda functions"
    objects = frozenset(n[len(OBJECT):] for n in _bound_names(base) if n.startswith(OBJECT))
    already = Counter(key for key, _, _ in _dead_code(base))
    for key, _, message in _dead_code(tree, objects):
        if already[key] > 0:
            already[key] -= 1  # the buggy file has this one too; a fault may have made it
            continue
        return message
    size, allowed = change_size(buggy, candidate), change_allowance(buggy)
    if size > allowed:
        return f"the change is too large for a localised repair ({size} syntax tokens; at most {allowed})"
    return None


def plan(buggy: str, original: str) -> tuple[int, list[dict]] | None:
    """(budget, minimal fix) when the minimal fix fits the budget and the gate, else None."""
    budget = budget_for(buggy)
    fix = minimal_fix(buggy, original)
    if sum(edit_cost(e) for e in fix) > budget or len(buggy) + len(original) <= budget:
        return None
    if structure_violation(buggy, original) is not None:
        return None
    current = buggy  # the oracle applies the fix edit by edit, and every step must pass the gate too
    for edit in fix:
        current = current.replace(edit["old"], edit["new"], 1)
        if structure_violation(buggy, current) is not None:
            return None
    return budget, fix
