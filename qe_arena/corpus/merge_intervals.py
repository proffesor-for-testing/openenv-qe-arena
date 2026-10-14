"""Merge closed integer intervals into a disjoint cover. Original."""

META = {
    "entry": "merge_intervals",
    "origin": "Original.",
    "tags": ["intervals", "invariants", "sorting"],
    "domain": "mathematics-or-formal-reasoning",
}


def merge_intervals(intervals, touching):
    """Merge closed integer intervals [lo, hi] into sorted, disjoint intervals covering the same integers.

    Intervals that overlap or share an endpoint always merge; when touching is True, intervals
    with no integer between them ([1, 3] and [4, 6]) merge as well. Returns
    {"merged": [[lo, hi], ...], "covered": c}, where c is the number of integers covered.
    Invariants, with gap = 1 if touching else 0, proved by induction over the intervals sorted by
    (lo, hi): (1) merged is sorted by lo; (2) consecutive a, b satisfy a.hi + gap < b.lo, because a
    sorted interval whose lo is at most last.hi + gap is absorbed into last and its hi kept with max,
    so no later interval can start further left; (3) the merged set equals the input set, since an
    interval is only absorbed into one it overlaps or touches; hence c = sum(hi - lo + 1).
    Raises ValueError if an interval does not have exactly two items or has lo > hi.
    """
    gap = 1 if touching else 0
    merged = []
    for lo, hi in sorted(intervals):
        if lo > hi:
            raise ValueError("an interval needs lo <= hi")
        if merged and lo <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    covered = sum(hi - lo + 1 for lo, hi in merged)
    return {"merged": merged, "covered": covered}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], False], [[[1, 3]], False], [[[1, 3], [4, 6]], False], [[[1, 3], [4, 6]], True],
        [[[1, 3], [5, 6]], True], [[[1, 3], [3, 6]], False], [[[4, 6], [1, 3]], True],
        [[[1, 10], [2, 3]], False], [[[2, 3], [1, 10], [11, 12]], True], [[[5, 5], [5, 5]], False],
        [[[0, 0], [2, 2], [4, 4]], False], [[[0, 0], [1, 1], [2, 2]], True], [[[3, 1]], False],
        [[[1, 2, 3]], False], [[[1]], True], [[[-5, -1], [0, 4]], True], [[[-5, -1], [0, 4]], False],
        [[[1, 4], [2, 3], [6, 8], [7, 9]], False], [[[1, 5], [2, 6], [3, 7]], True],
    ]
    while len(cases) < n:
        items = []
        for _ in range(rng.randint(0, 7)):
            lo = rng.randint(-20, 30)
            items.append([lo, lo + rng.randint(0, 6)])
        cases.append([items, rng.random() < 0.5])
    return cases[:n]
