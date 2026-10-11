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

OPERATORS = (
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


def enumerate_sites(source: str) -> list[Site]:
    """All single-token mutation sites in ``source``, ordered by position, operator, replacement.

    Each returned site, applied alone, yields source that parses and whose AST differs from the
    original's. Sites inside f-strings, decorators and annotations are skipped.
    """
    finder = _Finder(source)
    finder.visit()
    baseline = ast.dump(finder.tree)
    unique: dict[tuple, Site] = {}
    for site in finder.sites:
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


def rename_locals(source: str, rng) -> str:
    """Rename function-local variables to other plausible names, deterministically per ``rng``.

    Parameters, function names, attributes, keyword arguments, imports, builtins and any name bound
    at module scope are left alone. Only names bound exclusively inside function bodies (assignment,
    loop/with/comprehension targets, walrus) are renamed, and every occurrence of such a name is
    renamed consistently, so the code is semantically identical. Formatting is preserved.
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
        return source
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
    edits = []
    for name, new in mapping.items():
        for occ in occurrences[name]:
            l, c, el, ec = text.node_span(occ)
            edits.append((text.offset(l, c), text.offset(el, ec), new))
    edits.sort()
    renamed = _splice(source, edits)
    check = ast.parse(renamed)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in mapping:
            node.id = mapping[node.id]
    if ast.dump(check) != ast.dump(tree):  # pragma: no cover - defensive invariant
        raise AssertionError("rename_locals changed program structure")
    return renamed
