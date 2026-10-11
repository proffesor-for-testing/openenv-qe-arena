"""Bucket a QE task's complexity from code-size signals. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/state_encoder.py (MIT)."""

META = {
    "entry": "complexity_bucket",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/state_encoder.py",
    "tags": ["scoring", "thresholds"],
}


def complexity_bucket(lines_of_code, cyclomatic, num_dependencies, num_tests):
    """Return "simple", "moderate" or "complex" for a task context.

    Points: lines_of_code > 500 adds 2, else > 100 adds 1; cyclomatic > 20 adds 2,
    else > 10 adds 1; num_dependencies > 10 adds 1; num_tests > 100 adds 1.
    A total of 4 or more is "complex", 2 or 3 is "moderate", 0 or 1 is "simple".
    Raises ValueError if any argument is negative.
    """
    if min(lines_of_code, cyclomatic, num_dependencies, num_tests) < 0:
        raise ValueError("signals must be non-negative")
    score = 0
    if lines_of_code > 500:
        score += 2
    elif lines_of_code > 100:
        score += 1
    if cyclomatic > 20:
        score += 2
    elif cyclomatic > 10:
        score += 1
    if num_dependencies > 10:
        score += 1
    if num_tests > 100:
        score += 1
    if score >= 4:
        return "complex"
    if score >= 2:
        return "moderate"
    return "simple"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 0, 0, 0],
        [100, 10, 10, 100],
        [101, 0, 0, 0],
        [101, 11, 0, 0],
        [500, 20, 0, 0],
        [501, 0, 0, 0],
        [501, 11, 0, 0],
        [0, 21, 0, 0],
        [0, 21, 11, 101],
        [0, 0, 11, 0],
        [0, 0, 11, 101],
        [0, 11, 0, 101],
        [501, 21, 0, 0],
        [101, 11, 11, 0],
        [101, 11, 11, 101],
        [101, 0, 0, 100],
        [101, 0, 10, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, -5],
    ]
    while len(cases) < n:
        cases.append([rng.choice([rng.randint(0, 700), rng.choice([100, 101, 500, 501])]),
                      rng.choice([rng.randint(0, 30), rng.choice([10, 11, 20, 21])]),
                      rng.randint(5, 15), rng.randint(80, 120)])
    return cases[:n]
