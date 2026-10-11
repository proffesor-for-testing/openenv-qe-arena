"""Compute a simplified maintainability index for a module. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:docs/reports/analyze_complexity.py (MIT)."""

import math

META = {
    "entry": "maintainability_index",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:docs/reports/analyze_complexity.py",
    "tags": ["code-metrics", "math"],
}


def maintainability_index(lines, avg_complexity, halstead_volume):
    """Return a simplified maintainability index in [0, 100], rounded to 2 decimals.

    If lines == 0 or avg_complexity == 0, returns 100.0. Otherwise
      mi = 171 - 5.2 * ln(halstead_volume + 1) - 0.23 * avg_complexity - 16.2 * ln(lines)
    clamped to [0.0, 100.0] and rounded to 2 decimals (ln is the natural logarithm).
    The result is always a float.
    Raises ValueError if any argument is negative.
    """
    if lines < 0 or avg_complexity < 0 or halstead_volume < 0:
        raise ValueError("metrics must be non-negative")
    if lines == 0 or avg_complexity == 0:
        return 100.0
    mi = 171 - 5.2 * math.log(halstead_volume + 1) - 0.23 * avg_complexity - 16.2 * math.log(lines)
    mi = max(0.0, min(100.0, mi))
    return round(mi, 2)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 5, 100],
        [10, 0, 100],
        [0, 0, 0],
        [1, 1, 0],
        [10, 2, 50],
        [100, 10, 1000],
        [500, 20, 5000],
        [5000, 80, 100000],
        [50, 3, 0],
        [3, 1, 2],
        [5000, 0, 100000],
        [-1, 1, 1],
        [10, 1, -1],
    ]
    while len(cases) < n:
        cases.append([rng.randint(0, 3000), rng.randint(0, 40), rng.randint(0, 50000)])
    return cases[:n]
