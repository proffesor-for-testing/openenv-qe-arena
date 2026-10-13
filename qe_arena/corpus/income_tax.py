"""Progressive income tax over marginal bands in integer cents. Original."""

META = {
    "entry": "income_tax",
    "origin": "Original.",
    "tags": ["tax", "brackets", "money"],
    "domain": "finance-economics",
}


def income_tax(income_cents, allowance_cents, brackets):
    """Compute progressive income tax; amounts are integer cents.

    taxable = max(0, income_cents - allowance_cents). brackets lists [upper_cents, rate_bp]; each
    upper but the last is an int above the previous one (the first above 0) and the last is None.
    A band runs from the previous upper (0 for the first) to its own upper; taxable income inside it
    is taxed at rate_bp / 10000. Tax is the exact sum, rounded half up to a cent once at the end.
    Returns [tax_cents, marginal_rate_bp], the rate of the band holding the last cent of taxable
    (0 when taxable is 0).
    Raises ValueError for a negative income or allowance, empty brackets, a bracket that breaks
    these rules, or a rate outside 0..10000.
    """
    if income_cents < 0 or allowance_cents < 0 or not brackets:
        raise ValueError("invalid income")
    taxable = max(0, income_cents - allowance_cents)
    lower, owed, marginal = 0, 0, 0
    for index, (upper, rate) in enumerate(brackets):
        last = index == len(brackets) - 1
        if rate < 0 or rate > 10000 or (upper is None) != last:
            raise ValueError("invalid bracket")
        if not last and upper <= lower:
            raise ValueError("bracket uppers must increase")
        top = taxable if last else min(taxable, upper)
        if top > lower:
            owed += (top - lower) * rate
            marginal = rate
        lower = upper
    return [(owed + 5000) // 10000, marginal]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    uk = [[3770000, 2000], [12514000, 4000], [None, 4500]]
    cases = [
        [0, 0, uk],
        [1257000, 1257000, uk],
        [1257100, 1257000, uk],
        [5027000, 1257000, uk],
        [5027100, 1257000, uk],
        [13771000, 1257000, uk],
        [13771100, 1257000, uk],
        [20000000, 0, uk],
        [1000, 2000, uk],
        [3, 0, [[None, 1500]]],
        [1, 0, [[None, 5000]]],
        [10, 0, [[5, 0], [None, 10000]]],
        [5, 0, [[5, 0], [None, 10000]]],
        [6, 0, [[5, 1000], [None, 3000]]],
        [100, 0, []],
        [-1, 0, uk],
        [100, -1, uk],
        [100, 0, [[50, 1000]]],
        [100, 0, [[None, 1000], [None, 2000]]],
        [100, 0, [[50, 1000], [50, 2000], [None, 3000]]],
        [100, 0, [[0, 1000], [None, 2000]]],
        [100, 0, [[50, -1], [None, 2000]]],
        [100, 0, [[50, 1000], [None, 10001]]],
        [100, 0, [[50, 1000], [None, 10000]]],
        [4999, 0, [[None, 1]]],
        [0, 0, []],
        [100, 0, [[None, 0], [None, 2000]]],
    ]
    while len(cases) < n:
        uppers = sorted(rng.sample(range(1000, 20000000, 1000), rng.randint(0, 3)))
        rates = sorted(rng.randint(0, 6000) for _ in range(len(uppers) + 1))
        brackets = [[u, r] for u, r in zip(uppers + [None], rates)]
        income = rng.choice([rng.randint(0, 25000000), rng.choice(uppers + [0]) + rng.choice([-1, 0, 1, 37])])
        cases.append([max(0, income), rng.choice([0, 0, 1257000, rng.randint(0, 2000000)]), brackets])
    return cases[:n]
