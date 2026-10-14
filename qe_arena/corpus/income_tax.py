"""Progressive income tax over marginal bands in integer cents. Original."""

META = {
    "entry": "income_tax",
    "origin": "Original.",
    "tags": ["tax", "brackets", "money"],
    "domain": "finance-economics",
}


def income_tax(income_cents, allowance_cents, bands, top_rate_bp):
    """Compute progressive income tax in integer cents.

    taxable = max(0, income_cents - allowance_cents). bands lists [upper_cents, rate_bp] with
    increasing uppers. A band runs from the previous upper (0 for the first) to its own, and
    taxable income inside it is taxed at rate_bp / 10000; income above the last upper (all of it
    when bands is empty) is taxed at top_rate_bp / 10000. The tax is the exact sum, rounded half
    up to a cent once at the end.
    Returns [tax_cents, marginal_rate_bp], the rate of the band holding the last cent of taxable
    (0 when taxable is 0).
    Raises ValueError if income_cents or allowance_cents is negative or any rate is outside 0..10000.
    """
    if income_cents < 0 or allowance_cents < 0:
        raise ValueError
    taxable = max(0, income_cents - allowance_cents)
    lower, owed, marginal = 0, 0, 0
    for upper, rate in bands + [[taxable, top_rate_bp]]:
        if rate < 0 or rate > 10000:
            raise ValueError
        top = min(taxable, upper)
        if top > lower:
            owed += (top - lower) * rate
            marginal = rate
        lower = max(lower, upper)
    return [(owed + 5000) // 10000, marginal]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    uk = [[3770000, 2000], [12514000, 4000]]
    cases = [
        [0, 0, uk, 4500],
        [1257000, 1257000, uk, 4500],
        [1257100, 1257000, uk, 4500],
        [5027000, 1257000, uk, 4500],
        [5027100, 1257000, uk, 4500],
        [13771000, 1257000, uk, 4500],
        [13771100, 1257000, uk, 4500],
        [20000000, 0, uk, 4500],
        [1000, 2000, uk, 4500],
        [3, 0, [], 1500],
        [1, 0, [], 5000],
        [10, 0, [[5, 0]], 10000],
        [5, 0, [[5, 0]], 10000],
        [6, 0, [[5, 1000]], 3000],
        [-1, 0, uk, 4500],
        [100, -1, uk, 4500],
        [100, 0, [[50, -1]], 2000],
        [100, 0, [[50, 1000]], 10001],
        [100, 0, [[50, 1000]], 10000],
        [0, 0, [[50, -1]], 1000],
        [0, 0, [[50, 1000]], 10001],
        [40, 0, [[50, 1000]], 10001],
        [4999, 0, [], 1],
    ]
    while len(cases) < n:
        scale = rng.choice([1, 1, 1000])
        uppers = [u * scale for u in sorted(rng.sample(range(1, 20000), rng.randint(0, 3)))]
        rates = sorted(rng.randint(0, 6000) for _ in range(len(uppers) + 1))
        bands = [[u, r] for u, r in zip(uppers, rates)]
        income = rng.choice([rng.randint(0, 25000) * scale, rng.choice(uppers + [0]) + rng.choice([-1, 0, 1, 37])])
        cases.append([max(0, income), rng.choice([0, 0, 1257 * scale, rng.randint(0, 2000) * scale]), bands, rates[-1]])
    return cases[:n]
