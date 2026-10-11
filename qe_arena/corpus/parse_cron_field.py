"""Expand one cron field into the sorted values it matches. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/cli/utils/workflow-parser.ts (MIT)."""

import re

META = {
    "entry": "parse_cron_field",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/cli/utils/workflow-parser.ts",
    "tags": ["parsing", "cron", "time-window"],
}


def parse_cron_field(field, minimum, maximum):
    """Expand one cron field into the sorted list of values it selects, or None if invalid.

    field is a comma-separated list of items; each item is "*", "a", "a-b", "*/s",
    "a-b/s" or "a/s" where a, b, s are decimal integers. "*" means start = minimum;
    "*-b" is invalid. The end is b when given, else maximum for "*" or when a step is
    given ("a/s" runs from a to maximum), else a. The step defaults to 1. An item is
    invalid (the whole result is None) if it does not match that syntax, if step <= 0,
    start < minimum, end > maximum or start > end. Values from all items are merged,
    deduplicated and sorted ascending.
    """
    values = set()
    for item in field.split(","):
        match = re.fullmatch(r"(\*|\d+)(?:-(\d+))?(?:/(\d+))?", item)
        if not match or (match.group(1) == "*" and match.group(2) is not None):
            return None
        start = minimum if match.group(1) == "*" else int(match.group(1))
        if match.group(2) is not None:
            end = int(match.group(2))
        elif match.group(1) == "*" or match.group(3) is not None:
            end = maximum
        else:
            end = start
        step = 1 if match.group(3) is None else int(match.group(3))
        if step <= 0 or start < minimum or end > maximum or start > end:
            return None
        values.update(range(start, end + 1, step))
    return sorted(values)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["*", 1, 12],
        ["*", 0, 6],
        ["*/5", 0, 23],
        ["*/0", 0, 23],
        ["5", 0, 23],
        ["0", 0, 23],
        ["0", 1, 12],
        ["12", 1, 12],
        ["13", 1, 12],
        ["3-5", 1, 12],
        ["5-3", 1, 12],
        ["5-5", 1, 12],
        ["1-12/4", 1, 12],
        ["2/3", 1, 12],
        ["10/3", 0, 23],
        ["*-5", 0, 23],
        ["1,3,3,2", 1, 12],
        ["1-3,2-4", 1, 12],
        ["", 0, 23],
        ["1,", 0, 23],
        ["a", 0, 23],
        ["0-23/23", 0, 23],
        ["22-23", 0, 23],
        ["22-24", 0, 23],
    ]
    while len(cases) < n:
        low, high = rng.choice([(0, 23), (1, 12), (0, 7), (1, 15)])
        def item():
            kind = rng.randint(0, 4)
            a, b = rng.randint(low - 1, high), rng.randint(low, high + 1)
            if kind == 0:
                return str(a)
            if kind == 1:
                return f"{a}-{b}"
            if kind == 2:
                return f"*/{rng.randint(0, 6)}"
            if kind == 3:
                return f"{a}-{b}/{rng.randint(1, 5)}"
            return f"{a}/{rng.randint(2, 8)}"
        cases.append([",".join(item() for _ in range(rng.randint(1, 3))), low, high])
    return cases[:n]
