"""Compare a test success rate against its baseline. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/services/pattern_analytics.py (MIT)."""

META = {
    "entry": "baseline_verdict",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/services/pattern_analytics.py",
    "tags": ["analytics", "thresholds", "regression"],
}


def baseline_verdict(current_success, baseline_success):
    """Classify the change of a test success rate (integer percent) versus a baseline.

    baseline_success None -> "no baseline". Otherwise with d = current - baseline:
      d > 10        -> "significantly improved"
      0 < d <= 10   -> "improved"
      -5 <= d <= 0  -> "maintained"
      d < -5        -> "degraded"
    Raises ValueError if a given rate is outside 0..100.
    """
    if baseline_success is None:
        return "no baseline"
    for rate in (current_success, baseline_success):
        if rate < 0 or rate > 100:
            raise ValueError("rates are percentages")
    if current_success > baseline_success + 10:
        return "significantly improved"
    if current_success > baseline_success:
        return "improved"
    if current_success >= baseline_success - 5:
        return "maintained"
    return "degraded"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [50, None],
        [0, 0],
        [61, 50],
        [60, 50],
        [59, 50],
        [51, 50],
        [50, 50],
        [46, 50],
        [45, 50],
        [44, 50],
        [100, 89],
        [100, 90],
        [0, 6],
        [0, 5],
        [101, 50],
        [50, -1],
        [-1, None],
    ]
    while len(cases) < n:
        base = rng.randint(0, 100)
        cases.append([max(0, min(100, base + rng.randint(-15, 15))), base])
    return cases[:n]
