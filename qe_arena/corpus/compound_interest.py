"""Grow a deposit by compound interest with banker's rounding of each period. Original."""

META = {
    "entry": "compound_interest",
    "origin": "Original.",
    "tags": ["interest", "rounding", "money"],
    "domain": "finance-economics",
}


def compound_interest(principal_cents, annual_rate_bp, periods_per_year, years):
    """Compound a deposit in integer cents, rounding each period's interest half to even.

    There are periods_per_year * years periods. Each period's interest is
    balance * annual_rate_bp / (10000 * periods_per_year) rounded to a whole cent, an exact half
    cent going to the even neighbour (banker's rounding), and is added to the balance.
    Returns [final_balance, total_interest, ties]; ties counts periods whose exact interest ended
    in exactly half a cent.
    Raises ValueError if periods_per_year is not 1, 2, 4 or 12, or if principal_cents < 0,
    annual_rate_bp < 0, or years is outside 0..50.
    """
    if periods_per_year not in (1, 2, 4, 12):
        raise ValueError("invalid frequency")
    if principal_cents < 0 or annual_rate_bp < 0 or years < 0 or years > 50:
        raise ValueError("invalid terms")
    denominator = 10000 * periods_per_year
    balance, ties = principal_cents, 0
    for _ in range(periods_per_year * years):
        whole, rest = divmod(balance * annual_rate_bp, denominator)
        if 2 * rest == denominator:
            ties += 1
            whole += whole % 2
        elif 2 * rest > denominator:
            whole += 1
        balance += whole
    return [balance, balance - principal_cents, ties]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [50, 10000, 1, 1],
        [150, 10000, 1, 1],
        [250, 10000, 1, 1],
        [149, 10000, 1, 1],
        [151, 10000, 1, 1],
        [1050, 1200, 12, 1],
        [1150, 1200, 12, 1],
        [100000, 500, 1, 0],
        [100000, 500, 1, 1],
        [100000, 500, 2, 1],
        [100000, 500, 4, 2],
        [100000, 500, 12, 50],
        [0, 500, 12, 10],
        [100000, 0, 12, 10],
        [100000, 500, 3, 1],
        [100000, 500, 365, 1],
        [-1, 500, 12, 1],
        [100000, -1, 12, 1],
        [100000, 500, 12, -1],
        [100000, 500, 12, 51],
        [350, 2000, 4, 3],
        [12345, 375, 2, 7],
    ]
    while len(cases) < n:
        periods = rng.choice([1, 2, 4, 12])
        rate = rng.choice([600, 1200, 2400, 10000, 400 * periods, rng.randint(1, 2500)])
        principal = rng.randint(0, 2000) * 100 + rng.choice([50, 50, rng.randint(0, 99)])
        cases.append([principal, rate, periods, rng.randint(0, 12)])
    return cases[:n]
