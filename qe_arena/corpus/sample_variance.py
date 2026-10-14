"""Exact variance of integer measurements as a reduced fraction. Original."""

from fractions import Fraction

META = {
    "entry": "sample_variance",
    "origin": "Original.",
    "tags": ["statistics", "measurement", "exact-arithmetic"],
    "domain": "natural-science",
}


def sample_variance(values, population):
    """Return the variance of integer measurements as an exact fraction [numerator, denominator].

    With n = len(values) and mean = sum(values) / n, the sum of squared deviations
    S = sum((x - mean) ** 2 for x in values) is divided by n when population is True and by
    n - 1 (Bessel's correction, the sample variance) when population is False. The result is in
    lowest terms with a positive denominator; a zero variance is [0, 1].
    Raises ValueError if values is empty, or if population is False and there is only one value.
    """
    n = len(values)
    dof = n if population else n - 1
    if dof < 1:
        raise ValueError("not enough measurements for this variance")
    mean = Fraction(sum(values), n)
    spread = sum((Fraction(x) - mean) ** 2 for x in values)
    variance = spread / dof
    return [variance.numerator, variance.denominator]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], True], [[], False], [[7], True], [[7], False], [[3, 3], False], [[3, 5], False],
        [[3, 5], True], [[1, 2, 3, 4], False], [[1, 2, 3, 4], True], [[-4, 0, 4], False],
        [[10, 10, 10, 13], False], [[0, 1], True], [[2, 4, 4, 4, 5, 5, 7, 9], True],
        [[2, 4, 4, 4, 5, 5, 7, 9], False], [[-3, 9], False], [[100, 101, 99, 100, 102], False],
    ]
    while len(cases) < n:
        centre = rng.randint(-50, 400)
        values = [centre + rng.randint(-9, 9) for _ in range(rng.randint(1, 9))]
        cases.append([values, rng.random() < 0.4])
    return cases[:n]
