"""Decide a machine's maintenance status from hours, starts and calendar days. Original."""

META = {
    "entry": "maintenance_status",
    "origin": "Original.",
    "tags": ["maintenance", "scheduling", "asset-management"],
    "domain": "industrial-physical-systems",
}


def maintenance_status(hours, starts, days, limits):
    """Decide whether a machine is due for service from its usage since the last service.

    hours, starts and days are counters since the last service; limits is [hour_limit,
    start_limit, day_limit]. Each counter's usage is counter * 1000 // its limit (per mille).
    The driver is the counter with the highest usage, a tie going to the earlier of "hours",
    "starts", "days". With the driver's usage u, status is "overdue" if u >= 1100, "due" if
    u >= 1000, "soon" if u >= 900, else "ok".
    Returns [status, driver, u, remaining], remaining = the driver's limit minus its counter,
    or 0 when that is negative.
    Raises ValueError if a counter is negative, limits does not have 3 items, or a limit is below 1.
    """
    counters = [hours, starts, days]
    if min(counters) < 0 or len(limits) != 3 or min(limits) < 1:
        raise ValueError
    shares = [count * 1000 // limit for count, limit in zip(counters, limits)]
    usage = max(shares)
    index = shares.index(usage)
    if usage >= 1100:
        status = "overdue"
    elif usage >= 1000:
        status = "due"
    elif usage >= 900:
        status = "soon"
    else:
        status = "ok"
    driver = ("hours", "starts", "days")[index]
    return [status, driver, usage, max(0, limits[index] - counters[index])]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    base = [1000, 200, 365]
    cases = [
        [0, 0, 0, base],
        [899, 0, 0, base],
        [900, 0, 0, base],
        [999, 0, 0, base],
        [1000, 0, 0, base],
        [1099, 0, 0, base],
        [1100, 0, 0, base],
        [500, 100, 0, base],
        [500, 101, 0, base],
        [500, 100, 183, base],
        [100, 50, 400, base],
        [2000, 10, 10, base],
        [1, 1, 1, [1, 1, 1]],
        [950, 190, 0, base],
        [950, 191, 0, base],
        [0, 0, 0, [1000, 200]],
        [0, 0, 0, [1000, 200, 365, 30]],
        [0, 0, 0, [1000, 0, 365]],
        [10, 0, 0, [0, 200, 365]],
        [-1, 0, 0, base],
        [0, -1, 5, base],
        [0, 0, -1, base],
    ]
    while len(cases) < n:
        limits = [rng.choice([250, 500, 1000, 4000]), rng.choice([50, 100, 200]), rng.choice([30, 90, 180, 365])]
        counters = []
        for limit in limits:
            percent = rng.choice([0, 50, 85, 90, 95, 100, 105, 110, 120, rng.randint(0, 130)])
            counters.append(max(0, limit * percent // 100 + rng.choice([0, 0, -1, 1])))
        cases.append(counters + [limits])
    return cases[:n]
