"""Corpus of small, pure QE functions used by the bug-fix and mutant-kill task families.

Each module in this package follows the corpus contract in docs/DESIGN.md: a one-line module
docstring, a ``META`` dict, the entry function plus any helpers, and ``gen_inputs(rng, n)``.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import random
from dataclasses import dataclass
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CorpusEntry:
    name: str
    entry: str
    origin: str
    tags: tuple[str, ...]
    source: str
    function_source: str
    summary: str


def _is_kept(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    return isinstance(node, ast.FunctionDef) and node.name != "gen_inputs"


def _start_line(node: ast.stmt) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def extract_function_source(text: str) -> str:
    """Return the imports plus every function except ``gen_inputs``, verbatim from ``text``.

    Consecutive kept top-level statements form one run, copied with their original
    formatting and comments. Runs are joined with two blank lines.
    """
    lines = text.splitlines()
    runs: list[list[ast.stmt]] = []
    previous_kept = False
    for node in ast.parse(text).body:
        kept = _is_kept(node)
        if kept and previous_kept:
            runs[-1].append(node)
        elif kept:
            runs.append([node])
        previous_kept = kept
    chunks = [
        "\n".join(lines[_start_line(run[0]) - 1 : run[-1].end_lineno])
        for run in runs
    ]
    return "\n\n\n".join(chunks) + "\n"


def _summary(text: str) -> str:
    docstring = ast.get_docstring(ast.parse(text)) or ""
    first_line = docstring.strip().splitlines()[0] if docstring.strip() else ""
    for marker in (" Adapted from ", " Ported from ", " Original."):
        if marker in first_line:
            first_line = first_line.split(marker, 1)[0]
    return first_line.strip()


def _load_entries() -> dict[str, CorpusEntry]:
    entries: dict[str, CorpusEntry] = {}
    names = sorted(
        info.name
        for info in pkgutil.iter_modules([str(_PACKAGE_DIR)])
        if not info.name.startswith("_")
    )
    for name in names:
        module = importlib.import_module(f"{__name__}.{name}")
        text = (_PACKAGE_DIR / f"{name}.py").read_text(encoding="utf-8")
        meta = module.META
        entries[name] = CorpusEntry(
            name=name,
            entry=meta["entry"],
            origin=meta["origin"],
            tags=tuple(meta["tags"]),
            source=text,
            function_source=extract_function_source(text),
            summary=_summary(text),
        )
    return entries


ENTRIES: dict[str, CorpusEntry] = _load_entries()


def _module(name: str):
    if name not in ENTRIES:
        raise KeyError(f"unknown corpus entry: {name}")
    return importlib.import_module(f"{__name__}.{name}")


def call_reference(name: str, args: list):
    """Call the original entry function directly. For tests only; the verifier runs it in a sandbox."""
    module = _module(name)
    return getattr(module, ENTRIES[name].entry)(*args)


def gen_inputs(name: str, seed: int, n: int) -> list[list]:
    """Deterministic argument lists for ``name``: boundary cases first, then seeded random ones."""
    return _module(name).gen_inputs(random.Random(seed), n)


__all__ = ["CorpusEntry", "ENTRIES", "call_reference", "extract_function_source", "gen_inputs"]
