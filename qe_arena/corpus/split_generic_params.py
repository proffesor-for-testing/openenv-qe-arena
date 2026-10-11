"""Split a parameter list on top-level commas, respecting generics. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/parsers/multi-language-parser.ts (MIT)."""

META = {
    "entry": "split_generic_params",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/parsers/multi-language-parser.ts",
    "tags": ["parsing", "brackets"],
}


def split_generic_params(params):
    """Split a function parameter list on commas that are not nested in <> or ().

    Depth starts at 0, increases on "<" or "(" and decreases on ">" or ")" with no
    clamping, so an unmatched ">" (as in "=>") or ")" makes it negative. Only a comma
    at depth exactly 0 ends the current parameter; all other characters (including
    commas at any other depth) are kept. Each parameter is stripped of surrounding
    whitespace before it is added. The trailing parameter is added only if it is
    non-empty after stripping; every parameter ended by a comma is kept, even "".
    """
    result = []
    depth = 0
    current = ""
    for ch in params:
        if ch == "<" or ch == "(":
            depth += 1
        elif ch == ">" or ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            result.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        result.append(current.strip())
    return result


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [""],
        ["   "],
        ["a"],
        ["a, b"],
        ["a,b,"],
        [",a"],
        ["a,,b"],
        ["Map<K, V> m, int n"],
        ["f: (a, b) => c, d"],
        ["List<Map<A, B>> x,y"],
        ["a>b, c"],
        ["x: Pair<(u, v), w>, y: T"],
        ["a ,  b  , c"],
    ]
    pieces = ["a", "b", "Map<K, V>", "(x, y)", "List<T>", ",", ", ", " ", "<", ">", "(", ")", "n: int"]
    while len(cases) < n:
        cases.append(["".join(rng.choice(pieces) for _ in range(rng.randint(1, 7)))[:40]])
    return cases[:n]
