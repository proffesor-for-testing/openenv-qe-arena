"""Reward an agent for finishing faster and cheaper than estimated. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/reward_calculator.py (MIT)."""

META = {
    "entry": "efficiency_reward",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/reward_calculator.py",
    "tags": ["reward", "clamping"],
    "domain": "software-engineering",
}


def efficiency_reward(actual_seconds, expected_seconds, actual_cost, estimated_cost):
    """Return [time_reward, cost_reward], each rounded to 2 decimals.

    time_reward: 0.0 if actual_seconds <= 0; otherwise
      (expected_seconds / actual_seconds - 1) * 50 clamped to [-50, 50].
    cost_reward: 0.0 if actual_cost <= 0; otherwise
      (estimated_cost / actual_cost - 1) * 50 clamped to [-25, 50].
    The clamps are inclusive: values outside are replaced by the nearest bound.
    Rounding is Python's round(x, 2): an exact half goes to the even digit, so
    actual_seconds 16 with expected_seconds 3 gives -40.625 -> -40.62.
    """
    if actual_seconds <= 0:
        time_reward = 0.0
    else:
        time_reward = (expected_seconds / actual_seconds - 1.0) * 50.0
        time_reward = max(-50.0, min(time_reward, 50.0))
    if actual_cost <= 0:
        cost_reward = 0.0
    else:
        cost_reward = (estimated_cost / actual_cost - 1.0) * 50.0
        cost_reward = max(-25.0, min(cost_reward, 50.0))
    return [round(time_reward, 2), round(cost_reward, 2)]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 10, 0, 10],
        [-5, 10, -1, 10],
        [10, 10, 10, 10],
        [5, 10, 5, 10],
        [4, 10, 4, 10],
        [20, 10, 20, 10],
        [30, 10, 30, 10],
        [11, 10, 11, 10],
        [8, 10, 8, 10],
        [100, 1, 100, 1],
        [1, 100, 1, 100],
        [40, 30, 3, 2],
        [7, 9, 15, 12],
    ]
    while len(cases) < n:
        cases.append([rng.randint(0, 40), rng.randint(1, 40), rng.randint(0, 40), rng.randint(1, 40)])
    return cases[:n]
