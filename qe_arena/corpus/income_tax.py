"""Progressive income tax over marginal bands in integer cents. Original."""

META = {
    "entry": "income_tax",
    "origin": "Original.",
    "tags": ["tax", "brackets", "money"],
    "domain": "finance-economics",
}


def income_tax(income_cents, allowance_cents, brackets):
    """Compute progressive income tax in integer cents.

    taxable = max(0, income_cents - allowance_cents). brackets lists [upper_cents, rate_bp] with
    increasing uppers, the last upper None (no limit). A band runs from the previous upper (0 for
    the first) to its own, and taxable income inside it is taxed at rate_bp / 10000. The tax is the
    exact sum, rounded half up to a cent once at the end.
    Returns [tax_cents, marginal_rate_bp], the rate of the band holding the last cent of taxable
    (0 when taxable is 0).
    Raises ValueError if income_cents or allowance_cents is negative or any rate is outside 0..10000.
    """
    if income_cents < 0 or allowance_cents < 0:
        raise ValueError
    taxable = max(0, income_cents - allowance_cents)
    lower, owed, marginal = 0, 0, 0
    for upper, rate in brackets:
        if rate < 0 or rate > 10000:
            raise ValueError
        top = taxable if upper is None else min(taxable, upper)
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
        [-1, 0, uk],
        [100, -1, uk],
        [100, 0, [[50, -1], [None, 2000]]],
        [100, 0, [[50, 1000], [None, 10001]]],
        [100, 0, [[50, 1000], [None, 10000]]],
        [0, 0, [[50, -1], [None, 1000]]],
        [0, 0, [[50, 1000], [None, 10001]]],
        [40, 0, [[50, 1000], [None, 10001]]],
        [4999, 0, [[None, 1]]],
    ]
    while len(cases) < n:
        scale = rng.choice([1, 1, 1000])
        uppers = [u * scale for u in sorted(rng.sample(range(1, 20000), rng.randint(0, 3)))]
        rates = sorted(rng.randint(0, 6000) for _ in range(len(uppers) + 1))
        brackets = [[u, r] for u, r in zip(uppers + [None], rates)]
        income = rng.choice([rng.randint(0, 25000) * scale, rng.choice(uppers + [0]) + rng.choice([-1, 0, 1, 37])])
        cases.append([max(0, income), rng.choice([0, 0, 1257 * scale, rng.randint(0, 2000) * scale]), brackets])
    return cases[:n]
