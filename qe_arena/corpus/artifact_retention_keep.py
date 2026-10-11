"""Decide whether a stored test artifact survives the retention policy. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/storage/utils/retention.py (MIT)."""

META = {
    "entry": "artifact_retention_keep",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/storage/utils/retention.py",
    "tags": ["storage", "time-window", "policy"],
}


def artifact_retention_keep(rank, keep_latest_n, created_day, retention_days, expires_day,
                            default_ttl_days, today):
    """Return True if an artifact should be kept. Days are integer day numbers.

    rank is the artifact's 0-based position among the newest artifacts (0 = newest),
    or None if unranked. Rules in order:
    1. rank is not None and rank < keep_latest_n -> True (always kept).
    2. retention_days is not None and today > created_day + retention_days -> False.
    3. expires_day is not None and today > expires_day -> False.
    4. retention_days and expires_day are both None and
       today > created_day + default_ttl_days -> False.
    5. Otherwise True. An artifact is still kept on its last day (today equal to the limit).
    """
    if rank is not None and rank < keep_latest_n:
        return True
    if retention_days is not None and today > created_day + retention_days:
        return False
    if expires_day is not None and today > expires_day:
        return False
    if retention_days is None and expires_day is None:
        if today > created_day + default_ttl_days:
            return False
    return True


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 1, 0, None, None, 30, 100],
        [1, 1, 0, None, None, 30, 100],
        [None, 5, 0, None, None, 30, 100],
        [None, 5, 0, None, None, 30, 30],
        [None, 5, 0, None, None, 30, 31],
        [None, 0, 10, 7, None, 30, 17],
        [None, 0, 10, 7, None, 30, 18],
        [None, 0, 10, 7, None, 3, 16],
        [None, 0, 10, None, 20, 3, 20],
        [None, 0, 10, None, 20, 3, 21],
        [None, 0, 10, None, 20, 300, 21],
        [None, 0, 10, 30, 20, 300, 21],
        [None, 0, 10, 5, 40, 30, 16],
        [2, 3, 0, 1, 1, 1, 50],
        [3, 3, 0, 1, 1, 1, 50],
        [0, 0, 0, None, None, 0, 1],
    ]
    while len(cases) < n:
        created = rng.randint(0, 50)
        cases.append([rng.choice([None, rng.randint(0, 5)]), rng.randint(0, 4), created,
                      rng.choice([None, rng.randint(0, 30)]),
                      rng.choice([None, created + rng.randint(0, 40)]),
                      rng.randint(1, 40), created + rng.randint(0, 60)])
    return cases[:n]
