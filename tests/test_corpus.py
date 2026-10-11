"""Contract checks for every corpus entry (docs/DESIGN.md, "Corpus contract")."""
from __future__ import annotations

import ast
import json
import random
import re
import time

import pytest

from qe_arena import corpus
from qe_arena.corpus import ENTRIES, call_reference, gen_inputs

ALLOWED_IMPORTS = {
    "math", "re", "itertools", "functools", "collections", "string",
    "bisect", "heapq", "statistics", "decimal", "fractions",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "help", "memoryview",
    "exit", "quit",
}
MAX_FILE_BYTES = 6000
SOURCE_REPOS = ("sentinel-api-testing", "lionagi-qe-fleet", "agentic-qe")
ORIGIN_RE = re.compile(
    r"^proffesor-for-testing/(?:%s)@[0-9a-f]{40}:[\w./-]+\.(?:py|ts)$" % "|".join(SOURCE_REPOS)
)
ALLOWED_EXCEPTIONS = (ValueError, TypeError, KeyError)
SEEDS = (0, 1, 7)
N_INPUTS = 64
MAX_CALL_SECONDS = 0.005

NAMES = sorted(ENTRIES)


def _is_json_value(value) -> bool:
    return json.loads(json.dumps(value)) == value


def _mutation_sites(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            count += len(node.ops)
        elif isinstance(node, (ast.BinOp, ast.BoolOp, ast.AugAssign)):
            count += 1
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            count += 1
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            count += 1
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            count += 1
    return count


def _call(name: str, args: list):
    try:
        return "ok", call_reference(name, json.loads(json.dumps(args)))
    except ALLOWED_EXCEPTIONS as exc:
        return "raises", type(exc).__name__


def test_corpus_size_and_provenance():
    assert len(ENTRIES) >= 40
    adapted = [e for e in ENTRIES.values() if e.origin != "Original."]
    assert len(adapted) >= 28


@pytest.mark.parametrize("name", NAMES)
def test_contract_fields(name):
    entry = ENTRIES[name]
    module = corpus._module(name)
    assert set(module.META) == {"entry", "origin", "tags"}
    assert entry.entry == module.META["entry"]
    assert callable(getattr(module, entry.entry))
    assert callable(module.gen_inputs)
    assert entry.tags and all(isinstance(tag, str) and tag for tag in entry.tags)
    assert entry.summary and "\n" not in entry.summary
    assert "proffesor-for-testing" not in entry.summary and entry.summary.endswith(".")
    assert ast.get_docstring(ast.parse(entry.source))
    entry_defs = [
        node for node in ast.parse(entry.function_source).body
        if isinstance(node, ast.FunctionDef) and node.name == entry.entry
    ]
    assert len(entry_defs) == 1
    spec = ast.get_docstring(entry_defs[0])
    assert spec and len(spec) >= 60, "the entry docstring is the model's spec"


@pytest.mark.parametrize("name", NAMES)
def test_origin_format(name):
    entry = ENTRIES[name]
    docstring = ast.get_docstring(ast.parse(entry.source))
    if entry.origin == "Original.":
        assert "Original." in docstring
    else:
        assert ORIGIN_RE.match(entry.origin), entry.origin
        assert f"{entry.origin} (MIT)" in docstring


@pytest.mark.parametrize("name", NAMES)
def test_function_source_is_gated_and_self_contained(name):
    entry = ENTRIES[name]
    tree = ast.parse(entry.function_source)
    top_level = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert entry.entry in top_level
    assert "gen_inputs" not in top_level
    assert "META" not in entry.function_source
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {alias.name.split(".")[0] for alias in node.names} <= ALLOWED_IMPORTS
        elif isinstance(node, ast.ImportFrom):
            assert node.module and node.module.split(".")[0] in ALLOWED_IMPORTS
        elif isinstance(node, ast.Name):
            assert node.id not in BANNED_NAMES, node.id
            assert not (node.id.startswith("__") and node.id.endswith("__")), node.id
        elif isinstance(node, ast.Attribute):
            assert not node.attr.startswith("_"), node.attr
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.Match,
                               ast.AsyncFunctionDef, ast.Await)):
            pytest.fail(f"{type(node).__name__} is not allowed in corpus code")
    assert len(entry.source.encode("utf-8")) <= MAX_FILE_BYTES
    # The extracted source must be exactly a slice of the file text.
    for chunk in entry.function_source.rstrip("\n").split("\n\n\n"):
        assert chunk in entry.source


@pytest.mark.parametrize("name", NAMES)
def test_function_source_line_count(name):
    lines = ENTRIES[name].function_source.rstrip("\n").splitlines()
    assert 12 <= len(lines) <= 45, len(lines)


@pytest.mark.parametrize("name", NAMES)
def test_mutation_sites(name):
    assert _mutation_sites(ast.parse(ENTRIES[name].function_source)) >= 6


@pytest.mark.parametrize("name", NAMES)
def test_gen_inputs_deterministic_and_exact_length(name):
    for n in (1, 5, N_INPUTS, 100):
        first = gen_inputs(name, 3, n)
        assert len(first) == n
        assert first == gen_inputs(name, 3, n)
    for seed in SEEDS:
        cases = gen_inputs(name, seed, N_INPUTS)
        assert all(isinstance(args, list) for args in cases)
        assert all(_is_json_value(args) for args in cases)
    # Boundary cases come first, so they do not depend on the seed.
    assert gen_inputs(name, 0, 3) == gen_inputs(name, 99, 3)


@pytest.mark.parametrize("name", NAMES)
def test_outputs_are_json_and_fast(name):
    outcomes = []
    for seed in SEEDS:
        for args in gen_inputs(name, seed, N_INPUTS):
            best = float("inf")
            for _ in range(3):
                start = time.perf_counter()
                result = _call(name, args)
                best = min(best, time.perf_counter() - start)
            assert best < MAX_CALL_SECONDS, (args, best)
            if result[0] == "ok":
                assert _is_json_value(result[1]), (args, result[1])
            outcomes.append(result)
    # Each entry must exercise more than one behaviour over its inputs.
    assert len({json.dumps(o, sort_keys=True) for o in outcomes}) >= 2


@pytest.mark.parametrize("name", NAMES)
def test_function_source_matches_module(name):
    """The model-visible source alone reproduces the reference outputs."""
    entry = ENTRIES[name]
    namespace: dict = {}
    exec(compile(entry.function_source, f"<{name}>", "exec"), namespace)
    fn = namespace[entry.entry]
    for args in gen_inputs(name, 0, N_INPUTS):
        try:
            got = ("ok", fn(*json.loads(json.dumps(args))))
        except ALLOWED_EXCEPTIONS as exc:
            got = ("raises", type(exc).__name__)
        assert got == _call(name, args), args


def _leaf_types(value, path="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_types(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_types(item, f"{path}[{index}]")
    elif value is not None:
        yield path, type(value).__name__


@pytest.mark.parametrize("name", NAMES)
def test_output_types_are_stable(name):
    """The verifier compares strictly (2 != 2.0, True != 1), so a position keeps one type."""
    seen: dict[str, set[str]] = {}
    for seed in SEEDS:
        for args in gen_inputs(name, seed, N_INPUTS):
            kind, value = _call(name, args)
            if kind == "ok":
                for path, type_name in _leaf_types(value):
                    seen.setdefault(path, set()).add(type_name)
    mixed = {path: types for path, types in seen.items() if len(types & {"bool", "int", "float"}) > 1}
    assert not mixed


@pytest.mark.parametrize("name", NAMES)
def test_pure_and_does_not_mutate_arguments(name):
    for args in gen_inputs(name, 1, 16):
        snapshot = json.dumps(args, sort_keys=True)
        first = _call(name, args)
        assert json.dumps(args, sort_keys=True) == snapshot
        assert _call(name, args) == first


def test_gen_inputs_uses_given_rng_only():
    name = NAMES[0]
    random.seed(12345)
    a = gen_inputs(name, 5, N_INPUTS)
    random.seed(54321)
    assert gen_inputs(name, 5, N_INPUTS) == a


def test_extract_function_source_keeps_formatting():
    text = (
        '"""Doc. Original."""\n\nimport re\n\nMETA = {}\n\n\n'
        "def helper(x):\n    # comment kept\n    return x\n\n\n"
        "def entry(x):\n    return helper(x)\n\n\n"
        "def gen_inputs(rng, n):\n    return []\n"
    )
    assert corpus.extract_function_source(text) == (
        "import re\n\n\n"
        "def helper(x):\n    # comment kept\n    return x\n\n\n"
        "def entry(x):\n    return helper(x)\n"
    )
