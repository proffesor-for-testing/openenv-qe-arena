"""Classify the flakiness pattern of repeated test runs. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/agents/flaky_test_hunter.py (MIT)."""

META = {
    "entry": "flakiness_pattern",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/agents/flaky_test_hunter.py",
    "tags": ["flaky-tests", "sequences"],
}


def flakiness_pattern(results):
    """Classify a sequence of test outcomes (True = passed) from repeated runs.

    Let n = len(results) and let the longest consecutive run of passes and of failures
    be max_pass and max_fail.
    - Empty list -> "UNKNOWN".
    - max_pass == n -> "STABLE_PASS"; max_fail == n -> "STABLE_FAIL".
    - max_pass or max_fail strictly greater than 70% of n (x * 10 > n * 7)
      -> "INTERMITTENT".
    - Otherwise -> "RANDOM".
    """
    if not results:
        return "UNKNOWN"
    passes = fails = max_pass = max_fail = 0
    for passed in results:
        if passed:
            passes += 1
            fails = 0
            max_pass = max(max_pass, passes)
        else:
            fails += 1
            passes = 0
            max_fail = max(max_fail, fails)
    total = len(results)
    if max_pass == total:
        return "STABLE_PASS"
    if max_fail == total:
        return "STABLE_FAIL"
    if max_pass * 10 > total * 7 or max_fail * 10 > total * 7:
        return "INTERMITTENT"
    return "RANDOM"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    T, F = True, False
    cases = [
        [[]],
        [[T]],
        [[F]],
        [[T, T, T]],
        [[F, F, F, F]],
        [[T, F]],
        [[T, T, T, F]],
        [[F, T, T, T]],
        [[T, T, T, T, T, T, T, F, F, F]],
        [[T, T, T, T, T, T, T, T, F, F]],
        [[F, F, F, F, F, F, F, F, T, T]],
        [[F, F, F, F, F, F, F, T, T, T]],
        [[T, F, T, F, T, F]],
        [[T, T, F, F, T, T]],
        [[F, T, F, F, F, F]],
        [[T, F, F, F, F, F, F, F, F]],
    ]
    while len(cases) < n:
        bias = rng.random()
        cases.append([[rng.random() < bias for _ in range(rng.randint(1, 12))]])
    return cases[:n]
