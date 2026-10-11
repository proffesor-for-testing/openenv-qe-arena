"""Compute a 95% Wilson score interval for a pass rate. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/chaos-resilience/diagnosis/scorer.ts (MIT)."""

import math

META = {
    "entry": "wilson_interval",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/chaos-resilience/diagnosis/scorer.ts",
    "tags": ["statistics", "confidence-interval"],
}


def wilson_interval(successes, total):
    """Return the 95% Wilson score interval [low, high] for successes out of total.

    With z = 1.96 and p = successes / total:
      denominator = 1 + z*z / total
      center = (p + z*z / (2 * total)) / denominator
      margin = (z / denominator) * sqrt(p * (1 - p) / total + z*z / (4 * total * total))
    low = max(0.0, center - margin), high = min(1.0, center + margin), each a float
    rounded to 4 decimals. total == 0 returns [0.0, 0.0].
    Raises ValueError if successes < 0 or successes > total.
    """
    if successes < 0 or successes > total:
        raise ValueError("successes must be within 0..total")
    if total == 0:
        return [0.0, 0.0]
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [[0, 0], [0, 1], [1, 1], [0, 10], [10, 10], [5, 10], [1, 2], [9, 10],
             [50, 100], [99, 100], [3, 7], [-1, 5], [6, 5]]
    while len(cases) < n:
        total = rng.randint(1, 200)
        cases.append([rng.randint(0, total), total])
    return cases[:n]
