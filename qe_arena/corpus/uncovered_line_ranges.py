"""Collapse uncovered line numbers into contiguous ranges. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/coverage-analysis/services/coverage-parser.ts (MIT)."""

META = {
    "entry": "uncovered_line_ranges",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/coverage-analysis/services/coverage-parser.ts",
    "tags": ["coverage", "ranges"],
    "domain": "software-engineering",
}


def uncovered_line_ranges(lines):
    """Group uncovered line numbers into inclusive [start, end] ranges.

    The input may be unsorted and contain duplicates; it is deduplicated and sorted
    first. Consecutive numbers (each exactly one more than the previous) form one range.
    A single isolated line n becomes [n, n]. An empty list returns [].
    Raises ValueError if any line number is less than 1.
    """
    if any(line < 1 for line in lines):
        raise ValueError("line numbers start at 1")
    ordered = sorted(set(lines))
    if not ordered:
        return []
    ranges = []
    start = prev = ordered[0]
    for line in ordered[1:]:
        if line == prev + 1:
            prev = line
        else:
            ranges.append([start, prev])
            start = prev = line
    ranges.append([start, prev])
    return ranges


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[]],
        [[1]],
        [[1, 2]],
        [[1, 3]],
        [[3, 2, 1]],
        [[5, 5, 6]],
        [[1, 2, 4, 5, 7]],
        [[10, 12, 11, 20, 21, 30]],
        [[2, 4, 6, 8]],
        [[0, 1]],
        [[1, 2, 3, 5, 6, 7, 9]],
    ]
    while len(cases) < n:
        cases.append([[rng.randint(1, 25) for _ in range(rng.randint(1, 12))]])
    return cases[:n]
