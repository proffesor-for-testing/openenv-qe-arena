"""Least-squares gain and offset calibration in exact arithmetic. Original."""

from fractions import Fraction

META = {
    "entry": "calibration_fit",
    "origin": "Original.",
    "tags": ["calibration", "regression", "exact-arithmetic"],
    "domain": "natural-science",
}


def calibration_fit(pairs, tolerance):
    """Fit reference = gain * raw + offset to [raw, reference] integer pairs by ordinary least squares.

    With n pairs and sums Sx = sum(raw), Sy = sum(reference), Sxx = sum(raw * raw) and
    Sxy = sum(raw * reference): gain = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx) and
    offset = (Sy - gain * Sx) / n, both exact. A pair's residual is
    gain * raw + offset - reference; the fit is accepted when every residual's absolute value is
    <= tolerance. Returns [[gain_num, gain_den], [offset_num, offset_den], accepted], the
    fractions in lowest terms with a positive denominator and accepted a bool.
    Raises ValueError if fewer than two distinct raw readings are given (including no pairs).
    """
    n = len(pairs)
    sx = sum(raw for raw, _ in pairs)
    sy = sum(ref for _, ref in pairs)
    sxx = sum(raw * raw for raw, _ in pairs)
    sxy = sum(raw * ref for raw, ref in pairs)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        raise ValueError("need at least two distinct raw readings")
    gain = Fraction(n * sxy - sx * sy, denominator)
    offset = (sy - gain * sx) / n
    worst = max(abs(gain * raw + offset - ref) for raw, ref in pairs)
    return [[gain.numerator, gain.denominator], [offset.numerator, offset.denominator], worst <= tolerance]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], 0], [[[5, 10]], 0], [[[5, 10], [5, 12]], 1], [[[0, 1], [10, 21]], 0],
        [[[0, 3], [1, 5], [2, 7]], 0], [[[0, 0], [1, 1], [2, 3]], 0], [[[0, 0], [1, 1], [2, 3]], 1],
        [[[1, 2], [2, 3], [3, 5], [4, 4]], 1], [[[-2, 7], [2, -1]], 0], [[[3, 4], [7, 4]], 0],
        [[[100, 251], [200, 498], [300, 752]], 2], [[[100, 251], [200, 498], [300, 752]], 1],
        [[[1, 10], [3, 2]], 5], [[[0, 5], [4, 9], [8, 11], [12, 18]], 1],
    ]
    while len(cases) < n:
        gain, offset = rng.randint(-3, 5), rng.randint(-20, 20)
        raws = [rng.randint(-10, 40) for _ in range(rng.randint(2, 4 if rng.random() < 0.6 else 7))]
        pairs = [[x, gain * x + offset + rng.randint(-2, 2)] for x in raws]
        cases.append([pairs, rng.randint(0, 3)])
    return cases[:n]
