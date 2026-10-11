"""Match a file path against a glob with *, ** and ? wildcards. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/io/file-reader.ts (MIT)."""

import re

META = {
    "entry": "glob_match",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/io/file-reader.ts",
    "tags": ["glob", "parsing", "paths"],
}


def glob_to_regex(pattern):
    """Translate a glob into a regular expression body (without anchors)."""
    regex = ""
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if pattern[i:i + 2] == "**":
            if pattern[i + 2:i + 3] == "/":
                regex += "(?:.*/)?"
                i += 3
            else:
                regex += ".*"
                i += 2
        elif char == "*":
            regex += "[^/]*"
            i += 1
        elif char == "?":
            regex += "[^/]"
            i += 1
        elif char in ".+^${}()|[]\\":
            regex += "\\" + char
            i += 1
        else:
            regex += char
            i += 1
    return regex


def glob_match(pattern, path):
    """Return True if the whole path matches the glob pattern.

    "**/" anywhere in the pattern matches zero or more whole directories (nothing, or
    any text ending in "/"), so "a/**/b" matches "a/b"; "**" elsewhere matches anything,
    including "/"; "*" matches any run of characters except "/"; "?" matches exactly
    one character except "/". Every other character matches itself literally.
    """
    return re.fullmatch(glob_to_regex(pattern), path, re.DOTALL) is not None


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["*.py", "a.py"],
        ["*.py", "src/a.py"],
        ["**/*.py", "a.py"],
        ["**/*.py", "src/pkg/a.py"],
        ["src/**", "src/a/b.ts"],
        ["src/**", "src"],
        ["src/**/test_*.py", "src/test_a.py"],
        ["src/**/test_*.py", "src/x/y/test_a.py"],
        ["a?c", "abc"],
        ["a?c", "a/c"],
        ["a?c", "ac"],
        ["a.b", "axb"],
        ["a.b", "a.b"],
        ["(x)+[y]", "(x)+[y]"],
        ["*", ""],
        ["**", "a/b/c"],
        ["*/*", "a/b"],
        ["*/*", "a/b/c"],
        ["$x^", "$x^"],
        ["a**b", "a/x/b"],
    ]
    parts = ["src", "tests", "*", "**", "a", "b.py", "?", "c.ts", "*.py"]
    names = ["src", "tests", "a", "bb", "b.py", "c.ts", "x.py"]
    while len(cases) < n:
        pattern = "/".join(rng.choice(parts) for _ in range(rng.randint(1, 3)))
        path = "/".join(rng.choice(names) for _ in range(rng.randint(1, 3)))
        cases.append([pattern, path])
    return cases[:n]
