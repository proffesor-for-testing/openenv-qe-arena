"""Deterministic, formatting-preserving AST mutation of corpus functions.

Every mutation site is one concrete textual replacement at an exact (line, column) span of the
original source. Columns are 0-based *character* offsets (the AST reports UTF-8 byte offsets; they
are converted here), lines are 1-based. Applying a site splices only its span, so comments,
whitespace and every other character of the source are untouched.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import io
import keyword
import re
import tokenize
from dataclasses import dataclass
from typing import Iterator, Sequence

CLASSIC_OPERATORS = (
    "cmp_boundary",
    "cmp_negate",
    "arith_swap",
    "const_off_by_one",
    "bool_swap",
    "not_drop",
    "slice_shift",
    "range_bound",
    "return_flip",
    "augassign_swap",
    "min_max_swap",
    "break_continue_swap",
)
# Shapes of real defects (HumanEvalFix/DebugBench taxonomy and RuVector PR #1125), not token flips.
REALISTIC_OPERATORS = (
    "guard_delete",      # missing logic: an early-exit guard is gone
    "arm_drop",          # missing logic: one if/elif arm of a dispatch is gone
    "member_drop",       # missing logic: an allowlist literal lost one member
    "condition_drop",    # missing logic: one operand of an and/or is gone
    "extra_condition",   # excess logic: a branch also requires a variable to be truthy
    "call_swap",         # function misuse: a sibling call with a compatible signature
    "var_swap",          # variable misuse: another in-scope variable of the same type
    "const_sibling",     # value misuse: another literal of the same type from the same function
)
OPERATORS = CLASSIC_OPERATORS + REALISTIC_OPERATORS


@dataclass(frozen=True)
class Site:
    operator: str
    lineno: int
    col: int
    end_lineno: int
    end_col: int
    original: str
    replacement: str
    description: str


# --------------------------------------------------------------------------- source geometry

_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


class _Text:
    """Line table that maps AST/tokenize positions to absolute string offsets."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = _LINE_RE.findall(source)
        self.starts = []
        pos = 0
        for line in self.lines:
            self.starts.append(pos)
            pos += len(line)
        self.starts.append(pos)

    def line(self, lineno: int) -> str:
        return self.lines[lineno - 1] if 0 < lineno <= len(self.lines) else ""

    def char_col(self, lineno: int, byte_col: int) -> int:
        return len(self.line(lineno).encode("utf-8")[:byte_col].decode("utf-8", "replace"))

    def offset(self, lineno: int, col: int) -> int:
        if lineno == len(self.lines) + 1 and col == 0:
            return len(self.source)
        if not 0 < lineno <= len(self.lines):
            raise ValueError(f"line {lineno} out of range")
        return self.starts[lineno - 1] + col

    def node_span(self, node: ast.AST) -> tuple[int, int, int, int]:
        return (
            node.lineno,
            self.char_col(node.lineno, node.col_offset),
            node.end_lineno,
            self.char_col(node.end_lineno, node.end_col_offset),
        )

    def segment(self, span: tuple[int, int, int, int]) -> str:
        return self.source[self.offset(span[0], span[1]):self.offset(span[2], span[3])]


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    keep = (tokenize.OP, tokenize.NAME)
    return [t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type in keep]


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _skipped_subtrees(tree: ast.AST) -> set[ast.AST]:
    """Nodes inside f-strings, decorators and annotations: never mutated."""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            roots.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            roots.extend(node.decorator_list)
            if getattr(node, "returns", None) is not None:
                roots.append(node.returns)
        if isinstance(node, ast.arg) and node.annotation is not None:
            roots.append(node.annotation)
        if isinstance(node, ast.AnnAssign):
            roots.append(node.annotation)
    skipped: set[ast.AST] = set()
    for root in roots:
        skipped.update(ast.walk(root))
    return skipped


# --------------------------------------------------------------------------- site discovery

_CMP_TEXT = {
    ast.Lt: ("<",), ast.LtE: ("<=",), ast.Gt: (">",), ast.GtE: (">=",),
    ast.Eq: ("==",), ast.NotEq: ("!=",),
    ast.In: ("in",), ast.NotIn: ("not", "in"), ast.Is: ("is",), ast.IsNot: ("is", "not"),
}
_CMP_SWAP = {
    ast.Lt: ("cmp_boundary", "<="), ast.LtE: ("cmp_boundary", "<"),
    ast.Gt: ("cmp_boundary", ">="), ast.GtE: ("cmp_boundary", ">"),
    ast.Eq: ("cmp_negate", "!="), ast.NotEq: ("cmp_negate", "=="),
    ast.In: ("cmp_negate", "not in"), ast.NotIn: ("cmp_negate", "in"),
    ast.Is: ("cmp_negate", "is not"), ast.IsNot: ("cmp_negate", "is"),
}
_ARITH_SWAP = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "//"), ast.FloorDiv: ("//", "*")}
_BOOL_SWAP = {ast.And: ("and", "or"), ast.Or: ("or", "and")}
_AUG_SWAP = {ast.Add: ("+=", "-="), ast.Sub: ("-=", "+=")}
# Expressions that bind at least as tightly as binary "+", so "e + 1" means "(e) + 1".
_TIGHT_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.MatMult, ast.Pow)
_TIGHT_EXPRS = (ast.Name, ast.Constant, ast.Attribute, ast.Subscript, ast.Call, ast.List, ast.Tuple,
                ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _is_int_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _is_str_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.JoinedStr) or (
        isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)))


def _is_range_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"


class _Finder:
    def __init__(self, source: str) -> None:
        self.text = _Text(source)
        self.tree = ast.parse(source)
        self.tokens = _tokens(source)
        self.parents = _parents(self.tree)
        self.skipped = _skipped_subtrees(self.tree)
        self.sites: list[Site] = []

    # -- helpers
    def _site(self, operator: str, span: tuple[int, int, int, int], replacement: str) -> None:
        original = self.text.segment(span)
        if original == replacement:
            return
        description = f"line {span[0]}: {original!r} -> {replacement!r} ({operator})"
        self.sites.append(Site(operator, span[0], span[1], span[2], span[3], original, replacement,
                               description))

    def _tokens_between(self, start: tuple[int, int], end: tuple[int, int]) -> list[tokenize.TokenInfo]:
        return [t for t in self.tokens if t.start >= start and t.end <= end and t.string not in "()"]

    def _op_span(self, left: ast.AST, right: ast.AST, expect: tuple[str, ...]):
        """Span of the operator token(s) strictly between two operand nodes."""
        _, _, ll, lc = self.text.node_span(left)
        rl, rc, _, _ = self.text.node_span(right)
        found = self._tokens_between((ll, lc), (rl, rc))
        if tuple(t.string for t in found) != expect:
            return None
        return (found[0].start[0], found[0].start[1], found[-1].end[0], found[-1].end[1])

    def _shifted(self, expr: ast.AST, operator: str) -> None:
        span = self.text.node_span(expr)
        text = self.text.segment(span)
        tight = isinstance(expr, _TIGHT_EXPRS) or (
            isinstance(expr, ast.BinOp) and isinstance(expr.op, _TIGHT_BINOPS)) or (
            isinstance(expr, ast.UnaryOp) and not isinstance(expr.op, ast.Not))
        base = text if tight else f"({text})"
        self._site(operator, span, f"{base} + 1")
        self._site(operator, span, f"{base} - 1")

    # -- operators
    def visit(self) -> None:
        for node in ast.walk(self.tree):
            if node in self.skipped:
                continue
            handler = getattr(self, "_" + type(node).__name__, None)
            if handler is not None:
                handler(node)

    def _Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        for i, op in enumerate(node.ops):
            span = self._op_span(operands[i], operands[i + 1], _CMP_TEXT[type(op)])
            if span is not None:
                operator, replacement = _CMP_SWAP[type(op)]
                self._site(operator, span, replacement)

    def _BinOp(self, node: ast.BinOp) -> None:
        swap = _ARITH_SWAP.get(type(node.op))
        if swap is None or _is_str_literal(node.left) or _is_str_literal(node.right):
            return
        span = self._op_span(node.left, node.right, (swap[0],))
        if span is not None:
            self._site("arith_swap", span, swap[1])

    def _AugAssign(self, node: ast.AugAssign) -> None:
        swap = _AUG_SWAP.get(type(node.op))
        if swap is None or _is_str_literal(node.value):
            return
        span = self._op_span(node.target, node.value, (swap[0],))
        if span is not None:
            self._site("augassign_swap", span, swap[1])

    def _BoolOp(self, node: ast.BoolOp) -> None:
        word, other = _BOOL_SWAP[type(node.op)]
        for left, right in zip(node.values, node.values[1:]):
            span = self._op_span(left, right, (word,))
            if span is not None:
                self._site("bool_swap", span, other)

    def _UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ast.Not):
            return
        line, col, _, _ = self.text.node_span(node)
        if self.text.line(line)[col:col + 3] != "not":
            return
        end = col + 3
        rest = self.text.line(line)
        while end < len(rest) and rest[end] in " \t":
            end += 1
        self._site("not_drop", (line, col, line, end), "")

    def _Constant(self, node: ast.Constant) -> None:
        if type(node.value) is not int:
            return
        parent = self.parents.get(node)
        if _is_range_call(parent) and len(parent.args) == 3 and parent.args[2] is node:
            return
        needs_parens = (
            (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Pow) and parent.left is node)
            or (isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node)
            or (isinstance(parent, ast.Call) and parent.func is node))
        span = self.text.node_span(node)
        for value in (node.value + 1, node.value - 1):
            text = str(value)
            if value < 0 and needs_parens:
                text = f"({text})"
            self._site("const_off_by_one", span, text)

    def _Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.slice, ast.Slice):
            return
        for bound in (node.slice.lower, node.slice.upper):
            if bound is not None and not _is_int_literal(bound) and bound not in self.skipped:
                self._shifted(bound, "slice_shift")

    def _Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("min", "max"):
            self._site("min_max_swap", self.text.node_span(func), "max" if func.id == "min" else "min")
        if _is_range_call(node) and 1 <= len(node.args) <= 3 and not node.keywords:
            stop = node.args[0] if len(node.args) == 1 else node.args[1]
            if not isinstance(stop, ast.Starred) and not _is_int_literal(stop):
                self._shifted(stop, "range_bound")

    def _Return(self, node: ast.Return) -> None:
        value = node.value
        if isinstance(value, ast.Constant) and type(value.value) is bool:
            self._site("return_flip", self.text.node_span(value), str(not value.value))

    def _Break(self, node: ast.Break) -> None:
        self._site("break_continue_swap", self.text.node_span(node), "continue")

    def _Continue(self, node: ast.Continue) -> None:
        self._site("break_continue_swap", self.text.node_span(node), "break")


# --------------------------------------------------------------------------- realistic operators

_EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)
# Sibling calls with a compatible signature. Method siblings are swapped on any receiver.
_NAME_SIBLINGS = {"any": ("all",), "all": ("any",), "round": ("int",)}
_ITER_SIBLINGS = {"sorted": ("reversed",), "reversed": ("sorted",)}  # only where the result is iterated
# Method siblings by receiver type. They are used only where the receiver's type is inferred, and only
# swaps that change values are listed: values/keys or remove/discard would mostly kill by raising.
_METHOD_SIBLINGS = {
    "str": {
        "startswith": ("endswith",), "endswith": ("startswith",),
        "lower": ("upper",), "upper": ("lower",),
        "strip": ("lstrip", "rstrip"), "lstrip": ("strip", "rstrip"), "rstrip": ("strip", "lstrip"),
        "find": ("rfind",), "rfind": ("find",), "index": ("rindex",), "rindex": ("index",),
        "split": ("rsplit",), "rsplit": ("split",),
        "partition": ("rpartition",), "rpartition": ("partition",),
        "isalpha": ("isalnum",), "isalnum": ("isalpha",), "isdigit": ("isalnum",),
        "islower": ("isupper",), "isupper": ("islower",),
    },
    "list": {"append": ("extend",), "extend": ("append",)},
}
_MODULE_SIBLINGS = {("math", "floor"): ("ceil",), ("math", "ceil"): ("floor",)}
_STR_METHODS = frozenset({"lower", "upper", "strip", "lstrip", "rstrip", "startswith", "endswith", "split",
                          "rsplit", "find", "rfind", "replace", "encode", "isdigit", "isalpha", "isalnum",
                          "partition", "rpartition", "splitlines", "title", "casefold", "zfill"})
_METHOD_RESULT = {
    **{m: "str" for m in ("lower", "upper", "strip", "lstrip", "rstrip", "join", "replace", "title",
                          "casefold", "zfill")},
    **{m: "int" for m in ("count", "find", "rfind", "index", "rindex")},
    **{m: "list" for m in ("split", "rsplit", "splitlines")},
    **{m: "bool" for m in ("startswith", "endswith", "isdigit", "isalpha", "isalnum", "islower", "isupper")},
}
_CALL_RESULT = {"len": "int", "ord": "int", "int": "int", "str": "str", "chr": "str", "repr": "str",
                "sorted": "list", "list": "list", "dict": "dict", "set": "set", "float": "float", "bool": "bool"}
_SWAP_TYPES = frozenset({"int", "str", "list", "dict", "set", "bool", "float"})
_LOOSE = (ast.BoolOp, ast.IfExp, ast.NamedExpr, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _iterable_value(node: ast.AST, function) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.ListComp)):
        return True
    return _expr_type(node, function.types if function is not None else {}) in ("list", "str")


def _paren(text: str, node: ast.AST) -> str:
    return f"({text})" if isinstance(node, _LOOSE) else text


class _Function:
    """Scope facts for one function: bindings, dominance between statements, simple types."""

    def __init__(self, func: ast.FunctionDef, parents: dict) -> None:
        self.func = func
        self.parents = parents
        args = func.args
        self.params = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
        self.params += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        self.stores: dict[str, list[ast.Name]] = {}
        self.loads: list[ast.Name] = []
        self.comp_names: set[str] = set()
        self.position: dict[ast.stmt, tuple[int, int]] = {}
        self._collect(func, in_comp=False)
        self.types = self._infer_types()

    def _collect(self, node: ast.AST, in_comp: bool) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                for i, stmt in enumerate(value):
                    self.position[stmt] = (id(value), i)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue  # a nested scope is analysed on its own
            comp = in_comp or isinstance(child, _COMPREHENSIONS)
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Store):
                    (self.comp_names.add(child.id) if comp else self.stores.setdefault(child.id, []).append(child))
                elif isinstance(child.ctx, ast.Load):
                    self.loads.append(child)
            self._collect(child, comp)

    def statement(self, node: ast.AST) -> ast.stmt:
        while not isinstance(node, ast.stmt):
            node = self.parents[node]
        return node

    def _ancestors(self, stmt: ast.stmt):
        node = stmt
        while node is not None and node is not self.func:
            if isinstance(node, ast.stmt):
                yield node
            node = self.parents.get(node)

    def dominates(self, binding: ast.Name, use: ast.Name) -> bool:
        """True if ``binding`` has certainly executed whenever ``use`` runs."""
        bound_at = self.statement(binding)
        ancestors = list(self._ancestors(self.statement(use)))
        if isinstance(bound_at, ast.For) and any(binding is n for n in ast.walk(bound_at.target)):
            # A loop variable exists only inside the body: the loop may run zero times.
            return any(self._inside(a, bound_at.body) for a in ancestors)
        container, index = self.position.get(bound_at, (None, -1))
        return any(self.position.get(a, (None, -1))[0] == container and self.position[a][1] > index
                   for a in ancestors)

    @staticmethod
    def _inside(stmt: ast.stmt, body: list) -> bool:
        return any(stmt is s for s in body)

    def bound_names(self) -> set[str]:
        return (set(self.stores) | set(self.params)) - self.comp_names

    # -- types
    def _infer_types(self) -> dict[str, str]:
        types: dict[str, str] = {}
        for _ in range(3):
            found: dict[str, set] = {}
            for name in self.params:
                found[name] = self._param_types(name, types)
            for name, nodes in self.stores.items():
                found.setdefault(name, set()).update(self._binding_type(n, types) for n in nodes)
            types = {n: next(iter(t)) for n, t in found.items() if len(t) == 1 and None not in t}
        return {n: t for n, t in types.items() if t in _SWAP_TYPES and n not in self.comp_names}

    def _binding_type(self, node: ast.Name, types: dict) -> str | None:
        parent = self.parents[node]
        if isinstance(parent, ast.Assign) and node in parent.targets:
            return _expr_type(parent.value, types)
        if isinstance(parent, ast.AugAssign):
            kind = _expr_type(parent.value, types)
            return kind if kind == types.get(node.id) or node.id not in types else None
        if isinstance(parent, ast.NamedExpr):
            return _expr_type(parent.value, types)
        if isinstance(parent, ast.For) and parent.target is node:
            if _is_range_call(parent.iter):
                return "int"
            return "str" if _expr_type(parent.iter, types) == "str" else None
        return None

    def _param_types(self, name: str, types: dict) -> set:
        kinds = set()
        for load in self.loads:
            if load.id != name:
                continue
            parent = self.parents[load]
            if isinstance(parent, ast.Attribute) and isinstance(self.parents.get(parent), ast.Call):
                if parent.attr in _STR_METHODS:
                    kinds.add("str")
                elif parent.attr in ("append", "extend", "sort", "pop"):
                    kinds.add("list")
                elif parent.attr in ("get", "items", "keys", "values"):
                    kinds.add("dict")
            elif _is_range_call(parent):
                kinds.add("int")
            elif isinstance(parent, (ast.Compare, ast.BinOp)):
                others = [parent.left, *parent.comparators] if isinstance(parent, ast.Compare) else [
                    parent.left, parent.right]
                for other in others:
                    if other is not load:
                        kind = _expr_type(other, types)
                        if kind in ("int", "str") and not (isinstance(parent, ast.BinOp) and kind == "str"
                                                          and not isinstance(parent.op, ast.Add)):
                            kinds.add(kind)
        return kinds if len(kinds) == 1 else {None}


def _expr_type(node: ast.AST, types: dict) -> str | None:
    if isinstance(node, ast.Constant):
        return {bool: "bool", int: "int", float: "float", str: "str"}.get(type(node.value))
    if isinstance(node, ast.JoinedStr):
        return "str"
    simple = {ast.List: "list", ast.ListComp: "list", ast.Dict: "dict", ast.DictComp: "dict",
              ast.Set: "set", ast.SetComp: "set", ast.Compare: "bool"}
    if type(node) in simple:
        return simple[type(node)]
    if isinstance(node, ast.Name):
        return types.get(node.id)
    if isinstance(node, ast.UnaryOp):
        return "bool" if isinstance(node.op, ast.Not) else _expr_type(node.operand, types)
    if isinstance(node, ast.BinOp):
        left, right = _expr_type(node.left, types), _expr_type(node.right, types)
        if isinstance(node.op, ast.Div) and {left, right} <= {"int", "float"}:
            return "float"
        if left == right and left in ("int", "float") and not isinstance(node.op, ast.MatMult):
            return left
        if left == right == "str" and isinstance(node.op, ast.Add):
            return "str"
        if isinstance(node.op, ast.Mult) and {left, right} == {"str", "int"}:
            return "str"
        if left == right == "list" and isinstance(node.op, ast.Add):
            return "list"
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        kind = _expr_type(node.value, types)
        return kind if kind in ("str", "list") else None
    if isinstance(node, ast.IfExp):
        a, b = _expr_type(node.body, types), _expr_type(node.orelse, types)
        return a if a == b else None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id == "round" and len(node.args) == 1:
                return "int"
            return _CALL_RESULT.get(node.func.id)
        if isinstance(node.func, ast.Attribute):
            return _METHOD_RESULT.get(node.func.attr)
    return None


class _Realistic:
    """Adds the REALISTIC_OPERATORS sites to a _Finder."""

    def __init__(self, finder: "_Finder") -> None:
        self.f = finder
        self.text = finder.text
        self.module_names = {n.name for n in finder.tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        self.module_names |= {(a.asname or a.name).split(".")[0] for n in finder.tree.body
                              if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        self.functions = [_Function(n, finder.parents) for n in ast.walk(finder.tree)
                          if isinstance(n, ast.FunctionDef)]
        self.owner: dict[ast.AST, _Function] = {}  # innermost enclosing function of every node
        for function in self.functions:  # ast.walk order: an inner function overwrites its outer one
            for node in ast.walk(function.func):
                self.owner[node] = function

    def visit(self) -> None:
        for node in ast.walk(self.f.tree):
            if node in self.f.skipped:
                continue
            handler = getattr(self, "_" + type(node).__name__, None)
            if handler is not None:
                handler(node)
        for function in self.functions:
            self._var_swap(function)
            self._const_sibling(function)

    # -- helpers
    def _whole_lines(self, node: ast.AST) -> bool:
        line, col, end_line, end_col = self.text.node_span(node)
        tail = self.text.line(end_line)[end_col:].strip()
        return not self.text.line(line)[:col].strip() and (not tail or tail.startswith("#"))

    def _keyword_at(self, node: ast.AST, word: str) -> bool:
        line, col, _, _ = self.text.node_span(node)
        return self.text.line(line)[col:col + len(word) + 1] in (word + " ", word + "(")

    def _clause_keyword_after(self, node: ast.AST, word: str):
        """First NAME token ``word`` after the end of ``node``: the next if/elif/else clause."""
        end = (node.end_lineno, self.text.node_span(node)[3])
        return next((t for t in self.f.tokens if t.type == tokenize.NAME and t.string == word and t.start >= end),
                    None)

    def _is_elif(self, node: ast.If) -> bool:
        parent = self.f.parents.get(node)
        return isinstance(parent, ast.If) and parent.orelse == [node] and self._keyword_at(node, "elif")

    # -- missing logic
    def _If(self, node: ast.If) -> None:
        self._extra_condition(node.test)
        if self._is_elif(node):
            self._drop_elif(node)
        elif not node.orelse and isinstance(node.body[-1], _EXITS) and self._whole_lines(node):
            self.f._site("guard_delete", (node.lineno, 0, node.end_lineno + 1, 0), "")
        elif len(node.orelse) == 1 and self._is_elif(node.orelse[0]):
            # Drop the leading arm: "if A: x / elif B: y" becomes "if B: y".
            nxt = node.orelse[0]
            line, col, _, _ = self.text.node_span(node)
            if self.text.line(line)[col:col + 3] == "if ":
                self.f._site("arm_drop", (line, col, nxt.lineno, self.text.node_span(nxt)[1] + 4), "if")

    def _drop_elif(self, node: ast.If) -> None:
        if not self._whole_lines(node.body[-1]) or self.text.line(node.lineno)[:node.col_offset].strip():
            return
        if not node.orelse:
            end = node.end_lineno + 1
        elif len(node.orelse) == 1 and self._is_elif(node.orelse[0]):
            end = node.orelse[0].lineno
        else:
            token = self._clause_keyword_after(node.body[-1], "else")
            if token is None:
                return
            end = token.start[0]
        self.f._site("arm_drop", (node.lineno, 0, end, 0), "")

    def _Compare(self, node: ast.Compare) -> None:
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, (ast.In, ast.NotIn)) or not isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                continue
            elts = comparator.elts
            span = self.text.node_span(comparator)
            literal = self.text.segment(span)
            if len(elts) < 2 or span[0] != span[2] or any(isinstance(e, ast.Starred) for e in elts):
                continue
            if literal[0] + literal[-1] not in ("()", "[]", "{}"):
                continue
            parts = [self.text.segment(self.text.node_span(e)) for e in elts]
            for i in range(len(parts)):
                rest = parts[:i] + parts[i + 1:]
                comma = "," if isinstance(comparator, ast.Tuple) and len(rest) == 1 else ""
                self.f._site("member_drop", span, literal[0] + ", ".join(rest) + comma + literal[-1])

    def _BoolOp(self, node: ast.BoolOp) -> None:
        word = " and " if isinstance(node.op, ast.And) else " or "
        span = self.text.node_span(node)
        texts = [_paren(self.text.segment(self.text.node_span(v)), v) for v in node.values]
        for i in range(len(texts)):
            self.f._site("condition_drop", span, word.join(texts[:i] + texts[i + 1:]))

    # -- excess logic
    def _extra_condition(self, test: ast.AST) -> None:
        if isinstance(test, ast.Name):
            return
        span = self.text.node_span(test)
        body = self.text.segment(span)
        if isinstance(test, (ast.IfExp, ast.NamedExpr, ast.Lambda)) or (
                isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or)):
            body = f"({body})"
        function = self.owner.get(test)
        bound = function.bound_names() if function is not None else set()
        seen = []
        for node in ast.walk(test):
            if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id in seen:
                continue
            parent = self.f.parents.get(node)
            if (isinstance(parent, ast.Call) and parent.func is node) or node.id not in bound:
                continue  # only the function's own parameters and locals: never a comprehension variable
            seen.append(node.id)
            self.f._site("extra_condition", span, f"{node.id} and {body}")

    def _While(self, node: ast.While) -> None:
        self._extra_condition(node.test)

    def _comprehension(self, node: ast.comprehension) -> None:
        for test in node.ifs:
            self._extra_condition(test)

    # -- function misuse
    def _Call(self, node: ast.Call) -> None:
        func = node.func
        parent = self.f.parents.get(node)
        iterated = (isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is node)
        if isinstance(func, ast.Name):
            siblings = _NAME_SIBLINGS.get(func.id, ())
            if func.id == "round" and (len(node.args) != 1 or node.keywords):
                siblings = ()
            if iterated:
                siblings += _ITER_SIBLINGS.get(func.id, ())
            for other in siblings:
                self.f._site("call_swap", self.text.node_span(func), other)
        elif isinstance(func, ast.Attribute):
            siblings = self._method_siblings(node, func)
            if func.attr in ("split", "rsplit") and len(node.args) < 2 and not any(
                    k.arg == "maxsplit" for k in node.keywords):
                return  # without maxsplit, split and rsplit agree
            line, col, end_line, end_col = self.text.node_span(func)
            if self.text.line(end_line)[end_col - len(func.attr):end_col] != func.attr:
                return
            for other in siblings:
                self.f._site("call_swap", (end_line, end_col - len(func.attr), end_line, end_col), other)

    def _method_siblings(self, node: ast.Call, func: ast.Attribute) -> tuple[str, ...]:
        receiver = func.value
        if isinstance(receiver, ast.Name) and receiver.id in self.module_names:
            return _MODULE_SIBLINGS.get((receiver.id, func.attr), ())  # re.split has no re.rsplit
        function = self.owner.get(node)
        kind = _expr_type(receiver, function.types if function is not None else {})
        siblings = _METHOD_SIBLINGS.get(kind, {}).get(func.attr, ())
        if func.attr == "append" and not (node.args and _iterable_value(node.args[0], function)):
            return ()  # extend(x) on a non-iterable only raises
        return siblings

    # -- variable misuse
    def _var_swap(self, function: _Function) -> None:
        types = function.types
        bound = function.bound_names()
        for use in function.loads:
            kind = types.get(use.id)
            if kind is None or use in self.f.skipped or use.id not in bound:
                continue
            parent = self.f.parents.get(use)
            if isinstance(parent, ast.Call) and parent.func is use:
                continue
            span = self.text.node_span(use)
            if self.text.segment(span) != use.id:
                continue
            for other in sorted(n for n, t in types.items() if t == kind and n != use.id and n in bound):
                if other in function.params or any(function.dominates(b, use) for b in function.stores.get(other, [])):
                    self.f._site("var_swap", span, other)

    # -- value misuse
    def _const_sibling(self, function: _Function) -> None:
        docstring = function.func.body[0] if ast.get_docstring(function.func) is not None else None
        constants: dict[str, list] = {"int": [], "str": []}
        for node in ast.walk(function.func):
            if not isinstance(node, ast.Constant) or node in self.f.skipped:
                continue
            if docstring is not None and node is docstring.value:
                continue
            parent = self.f.parents.get(node)
            if _is_range_call(parent) and len(parent.args) == 3 and parent.args[2] is node:
                continue
            kind = "int" if type(node.value) is int else "str" if type(node.value) is str else None
            if kind is not None:
                constants[kind].append(node)
        for nodes in constants.values():
            first: dict = {}
            for node in nodes:
                first.setdefault(node.value, self.text.segment(self.text.node_span(node)))
            for node in nodes:
                for value, text in first.items():
                    if value != node.value:
                        self.f._site("const_sibling", self.text.node_span(node), text)


def enumerate_sites(source: str, operators: Sequence[str] = OPERATORS) -> list[Site]:
    """Mutation sites in ``source`` for ``operators``, ordered by position, operator, replacement.

    Each returned site, applied alone, yields source that parses and whose AST differs from the
    original's. Sites inside f-strings, decorators and annotations are skipped. Classic operators
    replace one token; realistic operators may replace an expression or delete whole lines.
    """
    unknown = set(operators) - set(OPERATORS)
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")
    finder = _Finder(source)
    finder.visit()
    if set(operators) & set(REALISTIC_OPERATORS):
        _Realistic(finder).visit()
    wanted = set(operators)
    baseline = ast.dump(finder.tree)
    unique: dict[tuple, Site] = {}
    for site in finder.sites:
        if site.operator in wanted:
            unique.setdefault((site.lineno, site.col, site.end_lineno, site.end_col, site.replacement), site)
    valid = []
    for site in unique.values():
        try:
            mutated = ast.parse(apply(source, [site]))
        except SyntaxError:
            continue
        if ast.dump(mutated) != baseline:
            valid.append(site)
    valid.sort(key=lambda s: (s.lineno, s.col, s.end_lineno, s.end_col, s.operator, s.replacement))
    return valid


# --------------------------------------------------------------------------- application

def _splice(source: str, edits: list[tuple[int, int, str]]) -> str:
    """Replace non-overlapping [start, end) ranges; edits must be sorted by start."""
    out, pos = [], 0
    for start, end, text in edits:
        out.append(source[pos:start])
        out.append(text)
        pos = end
    out.append(source[pos:])
    return "".join(out)


def apply(source: str, sites: Sequence[Site]) -> str:
    """Apply non-overlapping sites to ``source``. Raises ValueError on overlap or a stale site."""
    text = _Text(source)
    edits = []
    for site in sites:
        start = text.offset(site.lineno, site.col)
        end = text.offset(site.end_lineno, site.end_col)
        if source[start:end] != site.original:
            raise ValueError(f"site does not match source: {site.description}")
        edits.append((start, end, site.replacement, site))
    edits.sort(key=lambda e: (e[0], e[1]))
    for (s1, e1, _, a), (s2, e2, _, b) in zip(edits, edits[1:]):
        if s2 < e1 or (s1 == s2 and e1 == e2):
            raise ValueError(f"overlapping sites: {a.description} / {b.description}")
    return _splice(source, [(s, e, r) for s, e, r, _ in edits])


def unified_diff(a: str, b: str, path: str) -> str:
    """Unified diff of two sources (2 lines of context), labelled a/<path> and b/<path>."""
    def lines(s: str) -> list[str]:
        out = s.splitlines(keepends=True)
        if out and not out[-1].endswith(("\n", "\r")):
            out[-1] += "\n"
        return out
    return "".join(difflib.unified_diff(lines(a), lines(b), f"a/{path}", f"b/{path}", n=2))


def block_site(original: str, variant: str, operator: str, label: str) -> Site | None:
    """One whole-line site that turns ``original`` into ``variant``, or None if they are equal.

    The span runs from the first to the last line that differs, so a multi-hunk change becomes a
    single replacement that ``apply`` can combine with other, non-overlapping sites.
    """
    a = original.splitlines(keepends=True)
    b = variant.splitlines(keepends=True)
    changed = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not changed:
        return None
    i1, i2, j1, j2 = changed[0][1], changed[-1][2], changed[0][3], changed[-1][4]
    return Site(operator, i1 + 1, 0, i2 + 1, 0, "".join(a[i1:i2]), "".join(b[j1:j2]),
                f"line {i1 + 1}: {label} ({operator})")


# --------------------------------------------------------------------------- procedural renaming

_WORDS = (
    "acc", "buf", "cur", "idx", "item", "val", "tmp", "res", "count", "total", "part", "piece", "node",
    "elem", "pos", "step", "seen", "out", "left", "right", "head", "tail", "chunk", "entry", "token",
    "mark", "span", "width", "depth", "level", "limit", "prev", "best", "cell", "slot", "probe",
    "bucket", "window", "cursor", "pivot", "group", "batch", "frame", "lane", "row", "col", "unit",
    "score", "weight", "label", "record", "sample", "point", "edge", "field", "block", "stack",
)
_RESERVED = frozenset(dir(builtins)) | frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)
_FUNC_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _walk_scoped(node: ast.AST, in_function: bool = False) -> Iterator[tuple[ast.AST, bool]]:
    yield node, in_function
    inner = in_function or isinstance(node, _FUNC_SCOPES)
    for child in ast.iter_child_nodes(node):
        yield from _walk_scoped(child, inner)


def _string_bound_names(tree: ast.AST) -> set[str]:
    """Identifiers bound through string fields (defs, params, aliases, except/match captures)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def local_renaming(source: str, rng) -> dict[str, str]:
    """Choose new names for the function-local variables of ``source``, deterministically per ``rng``.

    Parameters, function names, attributes, keyword arguments, imports, builtins and any name bound
    at module scope are left alone. Only names bound exclusively inside function bodies (assignment,
    loop/with/comprehension targets, walrus) are renamed. New names never collide with any name
    already in ``source``. Returns {old: new}; empty when nothing can be renamed.
    """
    text = _Text(source)
    tree = ast.parse(source)
    excluded = _string_bound_names(tree) | _RESERVED
    in_fstring = {n.id for f in ast.walk(tree) if isinstance(f, ast.JoinedStr)
                  for n in ast.walk(f) if isinstance(n, ast.Name)}
    excluded |= in_fstring
    local_bound: set[str] = set()
    module_bound: set[str] = set()
    occurrences: dict[str, list[ast.Name]] = {}
    for node, in_function in _walk_scoped(tree):
        if not isinstance(node, ast.Name):
            continue
        occurrences.setdefault(node.id, []).append(node)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            (local_bound if in_function else module_bound).add(node.id)
    candidates = []
    for name in sorted(local_bound - module_bound - excluded,
                       key=lambda n: min((o.lineno, o.col_offset) for o in occurrences[n])):
        if name.startswith("__"):
            continue
        if all(text.segment(text.node_span(o)) == name for o in occurrences[name]):
            candidates.append(name)
    if not candidates:
        return {}
    taken = {t.string for t in _tokens(source) if t.type == tokenize.NAME} | _RESERVED
    pool = list(_WORDS)
    rng.shuffle(pool)
    mapping: dict[str, str] = {}
    suffix = 2
    for name in candidates:
        while not pool:
            pool = [f"{w}{suffix}" for w in _WORDS]
            suffix += 1
        new = pool.pop(0)
        while new in taken:
            if not pool:
                pool = [f"{w}{suffix}" for w in _WORDS]
                suffix += 1
            new = pool.pop(0)
        taken.add(new)
        mapping[name] = new
    return mapping


def rename_with(source: str, mapping: dict[str, str]) -> str:
    """Rename every ``ast.Name`` whose id is in ``mapping``, preserving formatting.

    Raises ValueError if a new name already occurs in ``source`` or if the rename would change
    the program's structure, so a mapping chosen for one source can be checked on a variant of it.
    """
    if not mapping:
        return source
    text = _Text(source)
    tree = ast.parse(source)
    present = {t.string for t in _tokens(source) if t.type == tokenize.NAME}
    if present & set(mapping.values()):
        raise ValueError(f"renaming collides with existing names: {sorted(present & set(mapping.values()))}")
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            span = text.node_span(node)
            if text.segment(span) != node.id:
                raise ValueError(f"cannot rename {node.id!r} at line {node.lineno}")
            edits.append((text.offset(span[0], span[1]), text.offset(span[2], span[3]), mapping[node.id]))
    edits.sort()
    renamed = _splice(source, edits)
    check = ast.parse(renamed)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            node.id = mapping[node.id]
    if ast.dump(check) != ast.dump(tree):
        raise ValueError("renaming changed program structure")
    return renamed


def rename_locals(source: str, rng) -> str:
    """Rename function-local variables to other plausible names, deterministically per ``rng``.

    Every occurrence of a renamed name changes consistently, so the code is semantically
    identical. See ``local_renaming`` for which names are eligible.
    """
    return rename_with(source, local_renaming(source, rng))
