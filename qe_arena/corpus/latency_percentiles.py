"""Summarize benchmark latencies with interpolated percentiles. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:scripts/comprehensive_benchmark.py (MIT)."""

META = {
    "entry": "latency_percentiles",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:scripts/comprehensive_benchmark.py",
    "tags": ["performance", "statistics"],
    "domain": "software-engineering",
}


def percentile(sorted_values, p):
    """Linear-interpolated percentile p (0-100) of a non-empty ascending list."""
    n = len(sorted_values)
    k = (n - 1) * p / 100
    f = int(k)
    c = f + 1 if f < n - 1 else f
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def latency_percentiles(times):
    """Summarize response times (integer ms) as a dict with keys min, max, p25, p50, p75, p95.

    The values are sorted ascending first. Percentile p uses linear interpolation on the
    sorted list v of length n: k = (n - 1) * p / 100, f = int(k), c = f + 1 unless f is
    the last index (then c = f); result = v[f] + (v[c] - v[f]) * (k - f).
    Percentiles are floats rounded to 2 decimals; min and max are returned as-is (ints).
    An empty list returns min and max 0 and every percentile 0.0.
    """
    if not times:
        return {"min": 0, "max": 0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    ordered = sorted(times)
    result = {"min": ordered[0], "max": ordered[-1]}
    for p in (25, 50, 75, 95):
        result["p" + str(p)] = round(float(percentile(ordered, p)), 2)
    return result


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[]],
        [[7]],
        [[10, 20]],
        [[30, 10, 20]],
        [[1, 2, 3, 4]],
        [[5, 1, 4, 2, 3]],
        [[100, 100, 100, 100, 100, 100]],
        [[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]],
        [[3, 9, 27, 81, 243, 729, 2187]],
        [[12, 15, 11, 19, 400, 13, 14, 16, 18, 17, 20, 10]],
    ]
    while len(cases) < n:
        cases.append([[rng.randint(1, 500) for _ in range(rng.randint(1, 12))]])
    return cases[:n]
