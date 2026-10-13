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
    anything with no variable in it, a bare builtin such as `zip` (when no name in the file shadows
    it), `x OP x`, `x or not x`, `x in (x,)`, `x or <truthy>` / `x and <falsy>`,
    `isinstance(_, object)` and `len(...) >= 0`. ``bound`` is every name the file binds.
    """
    if isinstance(test, ast.Name) and test.id in _BUILTINS and test.id not in bound:
        return True  # a builtin function object is always truthy
    if isinstance(test, (ast.Set, ast.List, ast.Tuple, ast.Dict)):
        return bool(test.keys if isinstance(test, ast.Dict) else test.elts)
    if isinstance(test, ast.JoinedStr):
        if any(isinstance(v, ast.Constant) and v.value for v in test.values):
            return True
        return None if any(isinstance(v, ast.FormattedValue) for v in test.values) else False
    if not any(isinstance(n, ast.Name) and (n.id not in _CONSTANT_NAMES or n.id in bound) for n in ast.walk(test)):
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
        if None in values:
            return None
        return FIXED if FIXED in values else not decisive
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if ast.dump(left) == ast.dump(right):
            if isinstance(op, (ast.In, ast.NotIn)):
                return None  # x in x is not fixed for lists
            return isinstance(op, (ast.Eq, ast.Is, ast.LtE, ast.GtE))
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set)) \
                and any(ast.dump(e) == ast.dump(left) for e in right.elts):
            return isinstance(op, ast.In)  # x in (x,) is always true
        if _is_len(left) and _int_value(right) is not None:
            limit = _int_value(right)
            if (isinstance(op, ast.GtE) and limit <= 0) or (isinstance(op, ast.Gt) and limit < 0):
                return True
        if _is_len(right) and _int_value(left) is not None:
            limit = _int_value(left)
            if (isinstance(op, ast.LtE) and limit <= 0) or (isinstance(op, ast.Lt) and limit < 0):
                return True
    if (isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "isinstance"
            and len(test.args) == 2 and isinstance(test.args[1], ast.Name) and test.args[1].id == "object"):
        return True
    return None


_BUILTINS = frozenset(n for n in dir(__import__("builtins")) if not n.startswith("_"))
_CONSTANT_NAMES = frozenset({"True", "False", "None", "object", "len", "int", "str", "bool", "float", "abs",
                             "min", "max", "sum", "round", "isinstance", "ord", "chr"})


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


def _bound_names(tree: ast.AST) -> frozenset:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
    return frozenset(names)


_RUNS = ("range", "zip", "sorted", "reversed", "list", "tuple", "enumerate", "str", "set", "map", "filter")


def _must_iterate(node: ast.AST, bound: frozenset) -> bool:
    """A for-loop iterable that is certainly non-empty: a non-empty literal, or a builtin over literals."""
    if _nonempty_literal(node):
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _RUNS
            and node.func.id not in bound and not node.keywords):
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
            return len(list(getattr(__import__("builtins"), node.func.id)(*args))) > 0
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


def _dead_code(tree: ast.AST) -> list[tuple[str, str, str]]:
    """Constant or tautological tests (a wrapper such as `if True:`), then unreachable statements.

    Each finding is (key, location, message). The key identifies the code itself, not its line,
    so findings the buggy file already has (a fault can create `x > x`) can be told apart from
    new ones.
    """
    found = []
    bound = _bound_names(tree)
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
    already = Counter(key for key, _, _ in _dead_code(base))
    for key, _, message in _dead_code(tree):
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
