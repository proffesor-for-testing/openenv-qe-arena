"""Tabulate a decaying activity in integer counts and find its half-life step. Original."""

META = {
    "entry": "decay_table",
    "origin": "Original.",
    "tags": ["simulation", "decay", "integer-arithmetic"],
    "domain": "natural-science",
}


def decay_table(activity, keep_num, keep_den, threshold, max_steps):
    """Tabulate a first-order decay (radioactive source or drug dose) one time step per row.

    Row 0 is [0, activity]. Each step multiplies the previous activity by keep_num / keep_den,
    the fraction retained per step, and rounds down to an integer count. Steps continue while
    the current activity is >= threshold and fewer than max_steps steps have been taken, so the
    first activity below threshold is the last row and no row has a step above max_steps.
    Returns {"rows": [[step, activity], ...], "half_life_step": h}, where h is the first step
    >= 1 whose activity a satisfies 2 * a <= the initial activity, or None if no row reaches it.
    Raises ValueError if activity < 0, threshold < 1, max_steps < 0, or unless
    0 <= keep_num < keep_den.
    """
    if activity < 0 or threshold < 1 or max_steps < 0:
        raise ValueError("bad table parameters")
    if not 0 <= keep_num < keep_den:
        raise ValueError("the retained fraction must be in [0, 1)")
    rows = [[0, activity]]
    half_life_step = None
    current, step = activity, 0
    while current >= threshold and step < max_steps:
        step += 1
        current = current * keep_num // keep_den
        rows.append([step, current])
        if half_life_step is None and 2 * current <= activity:
            half_life_step = step
    return {"rows": rows, "half_life_step": half_life_step}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [1000, 1, 2, 100, 20], [1000, 9, 10, 1, 0], [1000, 9, 10, 1, 1], [1000, 9, 10, 500, 30],
        [0, 1, 2, 1, 5], [-1, 1, 2, 1, 5], [100, 1, 2, 0, 5], [100, 1, 2, 1, -1],
        [100, 0, 3, 1, 5], [100, 3, 3, 1, 5], [100, -1, 3, 1, 5], [100, 4, 3, 1, 5],
        [64, 1, 2, 1, 10], [64, 1, 2, 64, 10], [64, 1, 2, 65, 10], [7, 2, 3, 2, 10],
        [10, 1, 2, 5, 10], [11, 1, 2, 6, 10], [1, 1, 2, 1, 3], [500, 7, 8, 100, 4],
    ]
    while len(cases) < n:
        den = rng.randint(2, 12)
        cases.append([rng.randint(0, 5000), rng.randint(0, den - 1), den, rng.randint(1, 300),
                      rng.randint(0, 25)])
    return cases[:n]
