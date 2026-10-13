"""Check expense claim lines against a travel-and-expense policy. Original."""

META = {
    "entry": "expense_violations",
    "origin": "Original.",
    "tags": ["expenses", "policy", "compliance"],
    "domain": "office-white-collar",
}


def expense_violations(items):
    """Return the policy violations in an expense claim as [index, code] pairs.

    items is a list of {"category": str, "cents": int, "receipt": bool, "date": str}.
    Items are checked in list order; for each item the codes are appended in this order:
    - "OVER_CAP": cents above the per-item cap for its category: "meal" 7500,
      "hotel" 25000, "taxi" 6000, "other" 10000 (exactly the cap is allowed);
    - "NO_RECEIPT": cents of 2500 or more and receipt is false;
    - "DAILY_MEALS": a meal for which the running total of meal cents on that same date
      (this item included, earlier items in the list only) is above 10000.
    Returns [] for a compliant claim. Raises ValueError for an unknown category or for
    cents of 0 or less; the claim is rejected at the first such item.
    """
    caps = {"meal": 7500, "hotel": 25000, "taxi": 6000, "other": 10000}
    meals_by_date = {}
    found = []
    for index, item in enumerate(items):
        category, cents = item["category"], item["cents"]
        if category not in caps:
            raise ValueError("unknown expense category: " + str(category))
        if cents <= 0:
            raise ValueError("expense amount must be positive")
        if cents > caps[category]:
            found.append([index, "OVER_CAP"])
        if cents >= 2500 and not item["receipt"]:
            found.append([index, "NO_RECEIPT"])
        if category == "meal":
            meals_by_date[item["date"]] = meals_by_date.get(item["date"], 0) + cents
            if meals_by_date[item["date"]] > 10000:
                found.append([index, "DAILY_MEALS"])
    return found


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def item(category, cents, receipt=True, date="2025-03-03"):
        return {"category": category, "cents": cents, "receipt": receipt, "date": date}

    cases = [
        [[]],
        [[item("meal", 7500), item("meal", 7501, date="2025-03-04")]],
        [[item("hotel", 25000), item("hotel", 25001)]],
        [[item("taxi", 6000), item("taxi", 6001)]],
        [[item("other", 10000), item("other", 10001)]],
        [[item("taxi", 2499, False), item("taxi", 2500, False), item("meal", 1, False)]],
        [[item("meal", 4000), item("meal", 6000), item("meal", 1, date="2025-03-04")]],
        [[item("meal", 4000), item("meal", 6001), item("meal", 500)]],
        [[item("meal", 3000, False), item("meal", 3000, date="2025-03-04"), item("hotel", 30000, False)]],
        [[item("taxi", 1200), item("meal", 0)]],
        [[item("taxi", 1200), item("parking", 900)]],
        [[item("meal", -5)]],
    ]
    categories = ["meal", "meal", "hotel", "taxi", "other"]
    dates = ["2025-03-03", "2025-03-04", "2025-03-05"]
    while len(cases) < n:
        claim = []
        for _ in range(rng.randint(1, 7)):
            cents = rng.choice([rng.randint(1, 4000), rng.randint(2000, 12000), rng.randint(5000, 32000)])
            claim.append(item(rng.choice(categories), cents, rng.random() < 0.7, rng.choice(dates)))
        cases.append([claim])
    return cases[:n]
