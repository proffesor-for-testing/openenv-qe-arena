"""Modular exponentiation by left-to-right square and multiply. Original."""

META = {
    "entry": "mod_pow",
    "origin": "Original.",
    "tags": ["number-theory", "modular-arithmetic", "exponentiation"],
    "domain": "mathematics-or-formal-reasoning",
}


def mod_pow(base, exponent, modulus):
    """Return base ** exponent % modulus, a value in [0, modulus), without building the power.

    Bits of exponent are read from the most significant one; for each bit the running result is
    squared modulo modulus, then multiplied by base modulo modulus when the bit is 1. Invariant:
    after the bits of the prefix p of exponent have been read, result == base ** p % modulus.
    base may be negative or larger than modulus. exponent 0 gives 1 % modulus, so 0 ** 0 is 1,
    and every result modulo 1 is 0.
    Raises ValueError if modulus < 1 or exponent < 0.
    """
    if modulus < 1 or exponent < 0:
        raise ValueError("need modulus >= 1 and exponent >= 0")
    result = 1 % modulus
    for bit in format(exponent, "b"):
        result = result * result % modulus
        if bit == "1":
            result = result * base % modulus
    return result


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [2, 10, 1000], [2, 10, 1], [0, 0, 7], [0, 0, 1], [5, 0, 13], [0, 5, 13], [3, 1, 7],
        [-2, 3, 7], [-2, 2, 7], [10, 3, 7], [7, 2, 7], [4, 13, 497], [2, 5, 0], [2, -1, 7],
        [3, 200, 1000000007], [-1, 9, 10], [-1, 8, 10], [2, 1, 2], [6, 7, 9], [2, 64, 18446744073],
    ]
    while len(cases) < n:
        cases.append([rng.randint(-50, 10 ** 6), rng.randint(0, 300), rng.randint(1, 10 ** 6)])
    return cases[:n]
