"""Reward a quality improvement plus the bugs it uncovered. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/reward_calculator.py (MIT)."""

META = {
    "entry": "quality_gain_reward",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/reward_calculator.py",
    "tags": ["reward", "quality-score"],
}


def quality_gain_reward(quality_before, quality_after, bugs_found, critical_bugs):
    """Return the integer quality reward for one QE action, capped at 100.

    A missing (None) quality score defaults to 50.
    reward = (quality_after - quality_before) * 2 + bugs_found * 5 + critical_bugs * 20,
    then min(reward, 100). Negative rewards are not floored. critical_bugs are scored in
    addition to bugs_found. Raises ValueError if bugs_found or critical_bugs is negative.
    """
    if bugs_found < 0 or critical_bugs < 0:
        raise ValueError("bug counts must be non-negative")
    before = 50 if quality_before is None else quality_before
    after = 50 if quality_after is None else quality_after
    reward = (after - before) * 2
    reward += bugs_found * 5
    reward += critical_bugs * 20
    return min(reward, 100)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [None, None, 0, 0],
        [None, 60, 0, 0],
        [40, None, 0, 0],
        [50, 50, 1, 0],
        [50, 50, 0, 1],
        [70, 60, 1, 1],
        [0, 50, 0, 0],
        [0, 50, 0, 1],
        [0, 50, 1, 0],
        [10, 100, 2, 1],
        [90, 10, 3, 2],
        [50, 50, -1, 0],
        [50, 50, 0, -1],
        [60, 61, 0, 4],
        [60, 61, 0, 5],
    ]
    while len(cases) < n:
        cases.append([rng.choice([None, rng.randint(0, 100)]), rng.choice([None, rng.randint(0, 100)]),
                      rng.randint(0, 6), rng.randint(0, 3)])
    return cases[:n]
