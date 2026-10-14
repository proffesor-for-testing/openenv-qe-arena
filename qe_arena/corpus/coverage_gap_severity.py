"""Rate the severity of an uncovered code range. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/coverage-analysis/services/coverage-parser.ts (MIT)."""

META = {
    "entry": "coverage_gap_severity",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/coverage-analysis/services/coverage-parser.ts",
    "tags": ["coverage", "thresholds"],
    "domain": "software-engineering",
}


def coverage_gap_severity(range_size, total_lines):
    """Return "critical", "high", "medium" or "low" for an uncovered range.

    percentage = range_size * 100 / total_lines (exact, not rounded).
    - percentage > 20 or range_size > 50 -> "critical"
    - percentage > 10 or range_size > 20 -> "high"
    - percentage > 5 or range_size > 10  -> "medium"
    - otherwise "low".
    Raises ValueError if total_lines < 1, range_size < 0 or range_size > total_lines.
    """
    if total_lines < 1 or range_size < 0 or range_size > total_lines:
        raise ValueError("invalid range")
    percentage = range_size * 100 / total_lines
    if percentage > 20 or range_size > 50:
        return "critical"
    if percentage > 10 or range_size > 20:
        return "high"
    if percentage > 5 or range_size > 10:
        return "medium"
    return "low"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 1],
        [1, 5],
        [1, 4],
        [10, 100],
        [11, 100],
        [20, 200],
        [21, 200],
        [5, 100],
        [6, 100],
        [50, 1000],
        [51, 1000],
        [20, 1000],
        [21, 1000],
        [10, 1000],
        [11, 1000],
        [21, 100],
        [5, 0],
        [-1, 10],
        [11, 10],
    ]
    while len(cases) < n:
        total = rng.choice([rng.randint(1, 100), rng.randint(100, 2000)])
        cases.append([rng.randint(0, min(total, 80)), total])
    return cases[:n]
