"""Format a duration in milliseconds for CLI output. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/cli/handlers/interfaces.ts (MIT)."""

META = {
    "entry": "format_duration",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/cli/handlers/interfaces.ts",
    "tags": ["formatting", "durations", "thresholds"],
}


def format_duration(ms):
    """Render a non-negative integer millisecond duration as a short string.

    ms < 1000 -> "<ms>ms"; ms < 60000 -> seconds with one decimal + "s";
    ms < 3600000 -> minutes with one decimal + "m"; otherwise hours with one decimal
    + "h". One decimal means format(value, ".1f"), e.g. 1500 -> "1.5s", 90000 -> "1.5m".
    Raises ValueError if ms is negative.
    """
    if ms < 0:
        raise ValueError("duration must be non-negative")
    if ms < 1000:
        return str(ms) + "ms"
    if ms < 60000:
        return format(ms / 1000, ".1f") + "s"
    if ms < 3600000:
        return format(ms / 60000, ".1f") + "m"
    return format(ms / 3600000, ".1f") + "h"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [[0], [999], [1000], [1500], [59999], [60000], [90000], [3599999],
             [3600000], [5400000], [86400000], [-1], [12345], [654321]]
    while len(cases) < n:
        cases.append([rng.choice([rng.randint(0, 2000), rng.randint(0, 120000), rng.randint(0, 20000000)])])
    return cases[:n]
