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


def _constant_test(test: ast.AST) -> bool:
    """A test whose value does not depend on the input: a literal, or x OP x such as `p == p`."""
    if isinstance(test, ast.Constant):
        return True
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and ast.dump(test.left) == ast.dump(test.comparators[0]):
        return True
    if isinstance(test, ast.UnaryOp):
        return _constant_test(test.operand)
    if isinstance(test, ast.BoolOp):
        return all(_constant_test(v) for v in test.values)
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


def _ends(block: list[ast.stmt], jumps: bool) -> bool:
    """True if control can never fall off the end of ``block``.

    ``jumps`` says whether break/continue end it (they do inside a loop body). The analysis is
    sound for exits it proves: a return, a raise, an if with every branch ending, a try whose
    body and handlers end or whose finally ends, a with whose body ends, and a loop that must run
    its body (a constant-true while, a for over a non-empty literal) with a body that ends and
    no break. A branch on a test that is not constant may go either way.
    """
    return any(_stmt_ends(stmt, jumps) for stmt in block)


def _stmt_ends(stmt: ast.stmt, jumps: bool) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, (ast.Break, ast.Continue)):
        return jumps
    if isinstance(stmt, ast.If):
        if _constant_test(stmt.test):
            return _ends(stmt.body, jumps) or _ends(stmt.orelse, jumps)  # either branch may be the live one
        return bool(stmt.orelse) and _ends(stmt.body, jumps) and _ends(stmt.orelse, jumps)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return _ends(stmt.body, jumps)
    if isinstance(stmt, ast.Try) or type(stmt).__name__ == "TryStar":
        if _ends(stmt.finalbody, jumps):
            return True
        handlers_end = all(_ends(h.body, jumps) for h in stmt.handlers)
        return handlers_end and (_ends(stmt.body, jumps) or (bool(stmt.orelse) and _ends(stmt.orelse, jumps)))
    if isinstance(stmt, ast.While):
        return _constant_test(stmt.test) and not _loop_body_jumps_out(stmt.body)
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return _nonempty_literal(stmt.iter) and _ends(stmt.body, False) and not _loop_body_jumps_out(stmt.body)
    return False


def _dead_code(tree: ast.AST) -> str | None:
    """A test whose value is fixed (a wrapper such as `if True:`), or an unreachable statement."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.IfExp)) and _constant_test(node.test):
            return f"a constant condition at line {node.lineno}"
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list) or not all(isinstance(s, ast.stmt) for s in block):
                continue
            for stmt, nxt in zip(block, block[1:]):
                if _stmt_ends(stmt, jumps=True):
                    return f"line {nxt.lineno} can never run (after line {stmt.lineno})"
        for handler in getattr(node, "handlers", []) or []:
            for stmt, nxt in zip(handler.body, handler.body[1:]):
                if _stmt_ends(stmt, jumps=True):
                    return f"line {nxt.lineno} can never run (after line {stmt.lineno})"
    return None


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
    dead = _dead_code(tree)
    if dead is not None and _dead_code(base) is None:
        return dead
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
    return budget, fix
