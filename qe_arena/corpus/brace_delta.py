"""Count the net curly-brace depth change of a source line, ignoring strings. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/parsers/multi-language-parser.ts (MIT)."""

META = {
    "entry": "brace_delta",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/parsers/multi-language-parser.ts",
    "tags": ["parsing", "strings"],
}


def brace_delta(line):
    """Return (number of "{") - (number of "}") in line, ignoring braces inside strings.

    A string starts at ", ' or ` and ends at the next occurrence of the same quote
    character that is not immediately preceded by a backslash. Braces inside a string
    are not counted. An unterminated string runs to the end of the line.
    """
    delta = 0
    in_string = False
    quote = ""
    for i in range(len(line)):
        ch = line[i]
        if in_string:
            if ch == quote and line[i - 1] != "\\":
                in_string = False
            continue
        if ch == '"' or ch == "'" or ch == "`":
            in_string = True
            quote = ch
        elif ch == "{":
            delta += 1
        elif ch == "}":
            delta -= 1
    return delta


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [""],
        ["{"],
        ["}"],
        ["{}"],
        ["if (x) {"],
        ["} else {"],
        ["'{'"],
        ['"}" + {'],
        ["`${a}` {"],
        ['"a\\"{" }'],
        ["'it\\'s {' {"],
        ['"{" \'}\' {'],
        ["'{"],
        ['"x" {{ "y" }'],
        ["{'\"'}"],
    ]
    alphabet = ['{', '}', '{', '}', '"', "'", '`', '\\', 'a', ' ']
    while len(cases) < n:
        cases.append(["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16)))])
    return cases[:n]
