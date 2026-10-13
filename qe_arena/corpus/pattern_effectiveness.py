"""Score how effective a reusable test-generation pattern has been. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/services/pattern_analytics.py (MIT)."""

META = {
    "entry": "pattern_effectiveness",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/services/pattern_analytics.py",
    "tags": ["scoring", "analytics"],
    "domain": "software-engineering",
}


def pattern_effectiveness(successful_generation, successful_tests, total_usage, avg_generation_ms):
    """Return a 0-100 effectiveness score rounded to 2 decimals.

    Raises ValueError if total_usage is negative, or if successful_generation or
    successful_tests is negative or greater than total_usage. Returns 0.0 when
    total_usage is 0. Otherwise:
      generation = successful_generation / total_usage * 40
      tests      = successful_tests / total_usage * 40
      speed      = min(20, 20 * 1000 / max(avg_generation_ms, 1))
    and the result is round(generation + tests + speed, 2).
    """
    if total_usage < 0:
        raise ValueError("total_usage must be non-negative")
    for count in (successful_generation, successful_tests):
        if count < 0 or count > total_usage:
            raise ValueError("success counts must be within [0, total_usage]")
    if total_usage == 0:
        return 0.0
    generation = successful_generation / total_usage * 40
    tests = successful_tests / total_usage * 40
    speed = min(20, 20 * 1000 / max(avg_generation_ms, 1))
    return round(generation + tests + speed, 2)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 0, 0, 500],
        [1, 0, 0, 500],
        [0, 0, -1, 500],
        [-1, 0, 4, 500],
        [0, 5, 4, 500],
        [4, 4, 4, 1000],
        [4, 4, 4, 999],
        [4, 4, 4, 1001],
        [2, 1, 4, 2000],
        [3, 1, 4, 4000],
        [1, 3, 8, 0],
        [1, 1, 3, 1],
        [0, 0, 5, 2500],
        [5, 0, 5, 3000],
        [0, 5, 5, 1500],
        [1, 2, 3, -50],
    ]
    while len(cases) < n:
        total = rng.randint(1, 50)
        cases.append([rng.randint(0, total), rng.randint(0, total), total, rng.randint(0, 5000)])
    return cases[:n]
