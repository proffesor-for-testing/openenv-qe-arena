"""Reward a flaky-test hunter with an F1 score minus miss penalties. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/reward_calculator.py (MIT)."""

META = {
    "entry": "flaky_hunter_reward",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/learning/reward_calculator.py",
    "tags": ["flaky-tests", "metrics", "reward"],
    "domain": "software-engineering",
}


def flaky_hunter_reward(true_positives, false_positives, false_negatives):
    """Reward for a flaky-test detector, rounded to 2 decimals.

    precision = tp / (tp + fp), or 0.0 when tp + fp == 0.
    recall    = tp / (tp + fn), or 0.0 when tp + fn == 0.
    f1 = 2 * precision * recall / (precision + recall), or 0.0 when both are 0.
    reward = f1 * 100 - 20 * false_negatives - 5 * false_positives (may be negative).
    Raises ValueError if any count is negative.
    """
    tp, fp, fn = true_positives, false_positives, false_negatives
    if min(tp, fp, fn) < 0:
        raise ValueError("counts must be non-negative")
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    reward = f1 * 100.0
    reward -= fn * 20.0
    reward -= fp * 5.0
    return round(reward, 2)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [3, 1, 0],
        [3, 0, 1],
        [2, 2, 2],
        [5, 1, 2],
        [1, 4, 0],
        [10, 0, 3],
        [-1, 0, 0],
        [0, 0, -2],
    ]
    while len(cases) < n:
        cases.append([rng.randint(0, 10), rng.randint(0, 4), rng.randint(0, 4)])
    return cases[:n]
