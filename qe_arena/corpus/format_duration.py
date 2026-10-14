"""Format a duration in milliseconds for CLI output. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/cli/handlers/interfaces.ts (MIT)."""

META = {
    "entry": "format_duration",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/cli/handlers/interfaces.ts",
    "tags": ["formatting", "durations", "thresholds"],
    "domain": "software-engineering",
}


def one_decimal(ms, unit):
    """ms / unit rounded half up to one decimal place, as text (exact integer arithmetic)."""
    tenths = (ms * 10 + unit // 2) // unit
    return str(tenths // 10) + "." + str(tenths % 10)


def format_duration(ms):
    """Render a non-negative integer millisecond duration as a short string.

    ms < 1000 -> "<ms>ms"; ms < 60000 -> seconds + "s"; ms < 3600000 -> minutes + "m";
    otherwise hours + "h". Seconds, minutes and hours have exactly one decimal, rounded
    half up from the exact value: 1500 -> "1.5s", 1450 -> "1.5s", 14350 -> "14.4s",
    59999 -> "60.0s", 90000 -> "1.5m". Raises ValueError if ms is negative.
    """
    if ms < 0:
        raise ValueError("duration must be non-negative")
    if ms < 1000:
        return str(ms) + "ms"
    if ms < 60000:
        return one_decimal(ms, 1000) + "s"
    if ms < 3600000:
        return one_decimal(ms, 60000) + "m"
    return one_decimal(ms, 3600000) + "h"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [[0], [999], [1000], [1500], [59999], [60000], [90000], [3599999],
             [3600000], [5400000], [86400000], [-1], [12345], [654321]]
    while len(cases) < n:
        cases.append([rng.choice([rng.randint(0, 2000), rng.randint(0, 120000), rng.randint(0, 20000000)])])
    return cases[:n]
