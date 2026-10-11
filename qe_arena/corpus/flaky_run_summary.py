"""Summarize repeated runs of one test into a flakiness verdict. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/agents/flaky_test_hunter.py (MIT)."""

META = {
    "entry": "flaky_run_summary",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/agents/flaky_test_hunter.py",
    "tags": ["flaky-tests", "statistics"],
}


def flaky_run_summary(results, iterations):
    """Summarize `iterations` runs of one test; results lists the outcomes seen (True = passed).

    Runs missing from results (when len(results) < iterations) count as failures.
    Returns {"passed", "failed", "is_flaky", "flakiness_pct", "pass_rate_pct"} where
    passed = number of True values, failed = iterations - passed,
    is_flaky = 0 < passed < iterations,
    flakiness_pct = min(passed, failed) * 200 // iterations (integer division, 0..100),
    pass_rate_pct = round(passed * 100 / iterations, 1).
    Raises ValueError if iterations < 1 or len(results) > iterations.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if len(results) > iterations:
        raise ValueError("more results than iterations")
    passed = sum(1 for r in results if r)
    failed = iterations - passed
    return {
        "passed": passed,
        "failed": failed,
        "is_flaky": 0 < passed < iterations,
        "flakiness_pct": min(passed, failed) * 200 // iterations,
        "pass_rate_pct": round(passed * 100 / iterations, 1),
    }


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    T, F = True, False
    cases = [
        [[], 1],
        [[T], 1],
        [[F], 1],
        [[T, F], 2],
        [[T, T], 2],
        [[T], 2],
        [[T, T, F], 3],
        [[T, F, F], 3],
        [[T, T, T, F, F, F, T], 7],
        [[T, T, T], 7],
        [[], 0],
        [[T, T], 1],
        [[T, F, T, T, T, T, T, T, T, T, T, T], 12],
    ]
    while len(cases) < n:
        k = rng.randint(1, 12)
        cases.append([[rng.random() < 0.6 for _ in range(rng.randint(0, k))], k])
    return cases[:n]
