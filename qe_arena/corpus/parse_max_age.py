"""Parse a max-age string such as "15m" into milliseconds. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/handlers/cross-phase-handlers.ts (MIT)."""

import re

META = {
    "entry": "parse_max_age",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/handlers/cross-phase-handlers.ts",
    "tags": ["parsing", "durations"],
}


def parse_max_age(max_age):
    """Convert "<digits><unit>" into milliseconds; return 0 for anything else.

    The whole string must match: one or more decimal digits followed by exactly one
    unit, "d" (days), "h" (hours), "m" (minutes) or "s" (seconds). Units are
    lowercase only and no whitespace is allowed. "0s" is 0; "90m" is 5400000.
    """
    match = re.fullmatch(r"(\d+)([dhms])", max_age)
    if not match:
        return 0
    num = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        return num * 24 * 60 * 60 * 1000
    if unit == "h":
        return num * 60 * 60 * 1000
    if unit == "m":
        return num * 60 * 1000
    return num * 1000


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [["0s"], ["1s"], ["1m"], ["1h"], ["1d"], ["90m"], ["2d"], ["36h"], ["7"], ["d"],
             ["1w"], ["1D"], [" 1h"], ["1h "], ["1.5h"], ["-1h"], ["10ms"], [""], ["007s"]]
    while len(cases) < n:
        cases.append([str(rng.randint(0, 120)) + rng.choice("dhmsdhmsx")])
    return cases[:n]
