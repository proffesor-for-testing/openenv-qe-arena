"""Ramp-limit an actuator setpoint once per control tick. Original."""

META = {
    "entry": "slew_limit",
    "origin": "Original.",
    "tags": ["control", "rate-limiter", "actuator"],
    "domain": "industrial-physical-systems",
}


def slew_limit(targets, start, max_rise, max_fall, deadband, low, high):
    """Ramp-limit an actuator setpoint, one output per control tick.

    Each target is clamped to [low, high]. If the clamped target is within deadband of the previous
    output (start before the first tick), abs(goal - previous) <= deadband, the output holds.
    Otherwise it moves toward the clamped target by at most max_rise per tick when rising and
    max_fall when falling, stopping exactly on the target when it is within reach.
    Returns the list of outputs, one per target.
    Raises ValueError if max_rise < 1, max_fall < 1, deadband < 0, or start is outside
    [low, high] (so also whenever low > high).
    """
    if max_rise < 1 or max_fall < 1 or deadband < 0 or not low <= start <= high:
        raise ValueError("invalid limiter settings")
    outputs, current = [], start
    for target in targets:
        goal = min(max(target, low), high)
        if abs(goal - current) <= deadband:
            pass
        elif goal > current:
            current = min(goal, current + max_rise)
        else:
            current = max(goal, current - max_fall)
        outputs.append(current)
    return outputs


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], 0, 5, 5, 0, 0, 100],
        [[10], 0, 5, 5, 0, 0, 100],
        [[10, 10, 10], 0, 5, 3, 0, 0, 100],
        [[0, 0, 0], 10, 5, 3, 0, 0, 100],
        [[4], 0, 5, 5, 0, 0, 100],
        [[5], 0, 5, 5, 0, 0, 100],
        [[6], 0, 5, 5, 0, 0, 100],
        [[3, 2], 0, 5, 5, 2, 0, 100],
        [[3, 2], 0, 5, 5, 3, 0, 100],
        [[150, 150, 150], 90, 7, 7, 0, 0, 100],
        [[-50, -50], 10, 7, 4, 0, 0, 100],
        [[50, 0, 50], 20, 10, 20, 1, 0, 100],
        [[7, 8, 9], 7, 1, 1, 0, 7, 9],
        [[10], 0, 0, 5, 0, 0, 100],
        [[10], 0, 5, 0, 0, 0, 100],
        [[10], 0, 5, 5, -1, 0, 100],
        [[10], 0, 5, 5, 0, 100, 0],
        [[10], 101, 5, 5, 0, 0, 100],
        [[10], -1, 5, 5, 0, 0, 100],
        [[5, 5], 5, 1, 1, 0, 5, 5],
        [[10], 0, 0, 5, 0, -10, 0],
        [[10], 0, 5, 5, 0, 1, 100],
        [[1], 5, 5, 5, 0, -10, 0],
        [[0], 1, 5, 5, 2, 0, 100],
        [[0], -5, 3, 3, 0, -10, 10],
    ]
    while len(cases) < n:
        low = rng.randint(-50, 50)
        high = low + rng.randint(0, 200)
        start = rng.randint(low, high)
        targets = [rng.randint(low - 30, high + 30) for _ in range(rng.randint(1, 8))]
        cases.append([targets, start, rng.randint(1, 40), rng.randint(1, 40), rng.choice([0, 0, 1, 3]), low, high])
    return cases[:n]
