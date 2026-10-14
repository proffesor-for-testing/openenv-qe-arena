"""Modular inverse by the extended Euclidean algorithm. Original."""

META = {
    "entry": "mod_inverse",
    "origin": "Original.",
    "tags": ["number-theory", "euclid", "modular-arithmetic"],
    "domain": "mathematics-or-formal-reasoning",
}


def extended_gcd(a, b):
    """Return [g, x, y] with g == gcd(a, b) and a * x + b * y == g, for a, b >= 0."""
    if b == 0:
        return [a, 1, 0]
    g, x, y = extended_gcd(b, a % b)
    return [g, y, x - (a // b) * y]


def mod_inverse(a, m):
    """Return the inverse of a modulo m: the unique x with 0 <= x < m and (a * x) % m == 1.

    a may be negative or at least m; only its residue a % m matters. Bezout coefficients come
    from the extended Euclidean algorithm: gcd(a, 0) == a == a * 1 + 0 * 0, and if
    b * x + (a % b) * y == g then a * y + b * (x - (a // b) * y) == g, because
    a % b == a - (a // b) * b. So with g == gcd(a % m, m) and (a % m) * x + m * y == g, the
    inverse exists exactly when g == 1, and it is x reduced into [0, m).
    Raises ValueError if m < 2 or gcd(a, m) != 1 (no inverse exists, including a % m == 0).
    """
    if m < 2:
        raise ValueError("the modulus must be at least 2")
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError("a is not invertible modulo m")
    return x % m


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [3, 7], [1, 2], [1, 5], [4, 5], [2, 4], [0, 7], [7, 7], [14, 7], [-3, 7], [10, 7],
        [3, 1], [3, 0], [3, -7], [17, 3120], [6, 9], [35, 64], [1, 1000003], [123456, 1000003],
        [-1, 10], [9, 10], [7, 26], [2, 3], [5, 12], [11, 12], [0, 1], [1, 3],
    ]
    while len(cases) < n:
        m = rng.randint(2, 5000)
        cases.append([rng.randint(-10000, 10000), m])
    return cases[:n]
