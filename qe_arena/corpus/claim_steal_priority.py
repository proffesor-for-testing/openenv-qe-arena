"""Score how urgently a stalled work claim should be stolen. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/coordination/claims/work-stealing.ts (MIT)."""

META = {
    "entry": "claim_steal_priority",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/coordination/claims/work-stealing.ts",
    "tags": ["scheduling", "scoring"],
    "domain": "software-engineering",
}


def claim_steal_priority(priority, stale_ms, time_to_deadline_ms, steal_count):
    """Return the integer steal priority of a work claim (higher = steal sooner).

    score = base + staleness + deadline - 10 * steal_count, where
    - base is 100 for "p0", 75 for "p1", 50 for "p2", 25 for "p3";
    - staleness = min(stale_ms // 60000, 50) (whole minutes since the last update);
    - deadline is 0 when time_to_deadline_ms is None, otherwise 50 if it is < 3600000
      (under an hour, including overdue negatives), 25 if < 86400000, else 0.
    Raises ValueError for an unknown priority or a negative stale_ms or steal_count.
    """
    base = {"p0": 100, "p1": 75, "p2": 50, "p3": 25}
    if priority not in base:
        raise ValueError("unknown priority")
    if stale_ms < 0 or steal_count < 0:
        raise ValueError("stale_ms and steal_count must be non-negative")
    score = base[priority] + min(stale_ms // 60000, 50)
    if time_to_deadline_ms is not None:
        if time_to_deadline_ms < 3600000:
            score += 50
        elif time_to_deadline_ms < 86400000:
            score += 25
    score -= steal_count * 10
    return score


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["p0", 0, None, 0],
        ["p1", 59999, None, 0],
        ["p2", 60000, None, 0],
        ["p3", 3000000, None, 0],
        ["p3", 3060000, None, 0],
        ["p3", 2999999, None, 0],
        ["p1", 0, 3599999, 0],
        ["p1", 0, 3600000, 0],
        ["p1", 0, 86399999, 0],
        ["p1", 0, 86400000, 0],
        ["p0", 0, -5000, 0],
        ["p2", 120000, 7200000, 2],
        ["p0", 0, None, 13],
        ["p4", 0, None, 0],
        ["P0", 0, None, 0],
        ["p1", -1, None, 0],
        ["p1", 0, None, -1],
    ]
    while len(cases) < n:
        cases.append([rng.choice(["p0", "p1", "p2", "p3"]), rng.randint(0, 4000000),
                      rng.choice([None, rng.randint(-100000, 100000000)]), rng.randint(0, 4)])
    return cases[:n]
