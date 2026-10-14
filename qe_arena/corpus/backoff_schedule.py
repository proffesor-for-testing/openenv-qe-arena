"""List capped exponential retry delays with their jitter allowance. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/retry-engine.ts (MIT)."""

META = {
    "entry": "backoff_schedule",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/shared/retry-engine.ts",
    "tags": ["retry", "backoff", "arithmetic"],
    "domain": "software-engineering",
}


def backoff_schedule(attempts, base_ms, max_ms, jitter_pct):
    """Return the maximum delay (ms) before each retry attempt 0..attempts-1.

    For attempt k: exponential = min(base_ms * 2**k, max_ms) and the delay is
    exponential + exponential * jitter_pct // 100 (the largest value the random jitter
    can add, using integer division). Returns a list of `attempts` ints ([] for 0).
    Raises ValueError if attempts < 0, base_ms < 1, max_ms < base_ms, or jitter_pct
    is outside 0..100.
    """
    if attempts < 0 or base_ms < 1 or max_ms < base_ms:
        raise ValueError("invalid backoff parameters")
    if jitter_pct < 0 or jitter_pct > 100:
        raise ValueError("jitter_pct must be within 0..100")
    delays = []
    for attempt in range(attempts):
        exponential = min(base_ms * 2 ** attempt, max_ms)
        delays.append(exponential + exponential * jitter_pct // 100)
    return delays


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 100, 1000, 0],
        [1, 100, 1000, 0],
        [4, 100, 1000, 0],
        [5, 100, 1000, 0],
        [6, 100, 1600, 25],
        [5, 1000, 32000, 25],
        [3, 7, 7, 50],
        [4, 3, 20, 33],
        [3, 10, 100, 100],
        [2, 10, 100, 101],
        [2, 10, 100, -1],
        [-1, 10, 100, 0],
        [2, 0, 100, 0],
        [2, 50, 49, 0],
        [10, 1, 1000, 10],
    ]
    while len(cases) < n:
        base = rng.randint(1, 500)
        cases.append([rng.randint(0, 10), base, base * rng.randint(1, 40), rng.randint(0, 100)])
    return cases[:n]
