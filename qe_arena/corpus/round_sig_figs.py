"""Round a fixed-point measurement to significant figures, ties to even. Original."""

META = {
    "entry": "round_sig_figs",
    "origin": "Original.",
    "tags": ["measurement", "rounding", "significant-figures"],
    "domain": "natural-science",
}


def round_sig_figs(value, scale, sig):
    """Round the measurement value * 10 ** (-scale) to sig significant figures.

    value is an integer count of the last recorded digit, so [12345, 3] is 12.345 and
    [12, -2] is 1200. Let d be the number of decimal digits of abs(value) (zero has one).
    If d <= sig the measurement is returned unchanged as [value, scale]. Otherwise the
    d - sig trailing digits of abs(value) are dropped, rounding half to even (12.345 -> 12.34 and
    12.355 -> 12.36 at 4 figures), and the sign of value is kept. When rounding carries into a new
    digit one more trailing digit is dropped, so the result still has exactly sig digits:
    999.5 at 3 figures, [9995, 1], becomes [100, -1], which is 1000. Returns
    [rounded, scale - dropped].
    Raises ValueError if sig < 1.
    """
    if sig < 1:
        raise ValueError("at least one significant figure is required")
    magnitude = abs(value)
    dropped = max(0, len(str(magnitude)) - sig)
    unit = 10 ** dropped
    kept, rest = divmod(magnitude, unit)
    if 2 * rest > unit or (2 * rest == unit and kept % 2 == 1):
        kept += 1
    if kept == 10 ** sig:
        kept //= 10
        dropped += 1
    return [-kept if value < 0 else kept, scale - dropped]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [12345, 3, 4], [12355, 3, 4], [12346, 3, 4], [12344, 3, 4], [-12345, 3, 4], [-12355, 3, 4],
        [9995, 1, 3], [99950, 2, 3], [9994, 1, 3], [-9996, 0, 3], [0, 2, 1], [7, 0, 1], [15, 0, 1],
        [25, 0, 1], [35, 0, 1], [45, -2, 1], [123, 0, 3], [123, 0, 5], [123, 0, 0], [1250, 2, 2],
        [1350, 2, 2], [1251, 2, 2], [999, 0, 2], [5, 0, 1], [-1, 0, 1], [1, 0, 1], [-1, 4, 3], [96, 0, 1], [1050, 0, 2],
    ]
    while len(cases) < n:
        sig = rng.randint(1, 5)
        value = rng.randint(1, 10 ** rng.randint(1, 8))
        if rng.random() < 0.5:
            value = value // 10 * 10 + 5
        cases.append([value * rng.choice([1, -1]), rng.randint(-3, 6), sig])
    return cases[:n]
