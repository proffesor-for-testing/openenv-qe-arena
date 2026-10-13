"""Check expense claim lines against a travel-and-expense policy. Original."""

META = {
    "entry": "expense_violations",
    "origin": "Original.",
    "tags": ["expenses", "policy", "compliance"],
    "domain": "office-white-collar",
}


def expense_violations(items):
    """Return one day's expense-policy violations as [index, code] pairs.

    items are [category, cents, receipt] lines, checked in order; a line adds its codes
    in this order: "OVER_CAP" if cents exceed the cap (meal 7500, hotel 25000, taxi 6000,
    other 10000); "NO_RECEIPT" if cents >= 2500 and receipt is false; "DAILY_MEALS" for a
    meal after which the day's meal total so far exceeds 10000. Raises ValueError at the
    first line with an unknown category or cents <= 0.
    """
    caps = {"meal": 7500, "hotel": 25000, "taxi": 6000, "other": 10000}
    meals, found = 0, []
    for index, (category, cents, receipt) in enumerate(items):
        if category not in caps:
            raise ValueError("unknown category")
        if cents <= 0:
            raise ValueError("cents must be positive")
        if cents > caps[category]:
            found.append([index, "OVER_CAP"])
        if cents >= 2500 and not receipt:
            found.append([index, "NO_RECEIPT"])
        if category == "meal":
            meals += cents
            if meals > 10000:
                found.append([index, "DAILY_MEALS"])
    return found


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[]],
        [[["meal", 7500, True]]],
        [[["meal", 7501, True]]],
        [[["hotel", 25000, True], ["hotel", 25001, True]]],
        [[["taxi", 6000, True], ["taxi", 6001, True]]],
        [[["other", 10000, True], ["other", 10001, True]]],
        [[["taxi", 2499, False], ["taxi", 2500, False]]],
        [[["meal", 1, False], ["meal", 9999, True]]],
        [[["meal", 4000, True], ["meal", 6001, True]]],
        [[["meal", 3000, False], ["hotel", 30000, False]]],
        [[["taxi", 1200, True], ["meal", 0, True]]],
        [[["taxi", 1200, True], ["bar", 900, True]]],
        [[["meal", -5, True]]],
    ]
    categories = ["meal", "meal", "hotel", "taxi", "other"]
    while len(cases) < n:
        claim = []
        for _ in range(rng.choice([1, 2, 2, 3])):
            cents = rng.choice([rng.randint(1, 4000), rng.randint(2000, 12000), rng.randint(5000, 32000)])
            claim.append([rng.choice(categories), cents, rng.random() < 0.7])
        cases.append([claim])
    return cases[:n]
