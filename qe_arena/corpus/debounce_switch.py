"""Debounce an analogue sensor into an on/off signal with hysteresis. Original."""

META = {
    "entry": "debounce_switch",
    "origin": "Original.",
    "tags": ["sensor", "debounce", "hysteresis"],
    "domain": "industrial-physical-systems",
}


def debounce_switch(samples, low, high, hold):
    """Turn sensor samples into a debounced on/off state with hysteresis.

    The state starts off. While off, it turns on once `hold` consecutive samples are >= high; while
    on, it turns off once `hold` consecutive samples are <= low. A sample that does not continue
    the run resets the count to zero, and the count restarts at zero after every change.
    Returns [final_state, changes]; changes lists [index, state] for every change, index being the
    position of the sample that completed the run.
    Raises ValueError if low >= high or hold < 1.
    """
    if low >= high or hold < 1:
        raise ValueError("invalid debounce settings")
    state, run, changes = False, 0, []
    for index, sample in enumerate(samples):
        crossing = sample <= low if state else sample >= high
        run = run + 1 if crossing else 0
        if run >= hold:
            state, run = not state, 0
            changes.append([index, state])
    return [state, changes]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], 10, 20, 1],
        [[20], 10, 20, 1],
        [[19], 10, 20, 1],
        [[21, 9], 10, 20, 1],
        [[21, 10], 10, 20, 1],
        [[21, 11], 10, 20, 1],
        [[25, 25], 10, 20, 2],
        [[25, 15, 25], 10, 20, 2],
        [[25, 25, 25, 5, 5], 10, 20, 2],
        [[25, 25, 5, 25, 5, 5], 10, 20, 2],
        [[25, 25, 25, 25], 10, 20, 3],
        [[30, 0, 30, 0], 10, 20, 1],
        [[15, 15, 15], 10, 20, 1],
        [[5, 5, 5], 10, 20, 1],
        [[25], 20, 20, 1],
        [[25], 21, 20, 1],
        [[25], 10, 20, 0],
        [[0, 100, 100, 100, 0, 0, 0, 0], 40, 60, 3],
        [[1], -5, 0, 0],
        [[1], 0, 5, 0],
    ]
    while len(cases) < n:
        low = rng.randint(0, 50)
        high = low + rng.randint(1, 30)
        samples = [rng.choice([rng.randint(low - 10, low), rng.randint(high, high + 10), rng.randint(low, high)])
                   for _ in range(rng.randint(1, 12))]
        cases.append([samples, low, high, rng.choice([1, 1, 2, 2, 3])])
    return cases[:n]
