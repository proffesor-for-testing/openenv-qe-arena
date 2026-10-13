"""Sliding-window login rate limiter with lockout. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/auth_service/main.py (MIT)."""

META = {
    "entry": "login_rate_limit",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/auth_service/main.py",
    "tags": ["security", "rate-limit", "time-window"],
    "domain": "software-engineering",
}


def login_rate_limit(request_times, lockout_until, now, limit, lockout_seconds):
    """Decide whether a login attempt at time `now` is rate limited.

    All times are integer seconds. Returns [limited, retry_after]:
    - If lockout_until is not None and now < lockout_until: [True, lockout_until - now].
    - Otherwise only previous requests with now - t < 60 count (a request exactly
      60 seconds old no longer counts). If the number of counted requests is
      >= limit, a new lockout starts: [True, lockout_seconds].
    - Otherwise [False, None].
    Raises ValueError if limit < 1.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if lockout_until is not None and now < lockout_until:
        return [True, lockout_until - now]
    recent = [t for t in request_times if now - t < 60]
    if len(recent) >= limit:
        return [True, lockout_seconds]
    return [False, None]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], None, 100, 1, 300],
        [[], 150, 100, 5, 300],
        [[], 100, 100, 5, 300],
        [[], 99, 100, 5, 300],
        [[99], 101, 100, 1, 300],
        [[40], None, 100, 1, 300],
        [[41], None, 100, 1, 300],
        [[41, 42], None, 100, 3, 300],
        [[41, 42, 43], None, 100, 3, 300],
        [[40, 41, 42], None, 100, 3, 300],
        [[10, 20, 95, 96], None, 100, 2, 120],
        [[10, 20, 95, 96], None, 100, 3, 120],
        [[1, 2, 3], 50, 100, 2, 60],
        [[1, 2, 3], None, 100, 0, 60],
        [[100, 100], None, 100, 2, 30],
        [[], None, 0, 1, 10],
    ]
    while len(cases) < n:
        now = rng.randint(60, 200)
        times = sorted(rng.randint(now - 90, now) for _ in range(rng.randint(0, 8)))
        lockout = rng.choice([None, None, rng.randint(now - 20, now + 20)])
        cases.append([times, lockout, now, rng.randint(1, 6), rng.choice([30, 60, 300])])
    return cases[:n]
