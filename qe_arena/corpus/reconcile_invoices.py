"""Match payments to invoices and classify each invoice's settlement. Original."""

META = {
    "entry": "reconcile_invoices",
    "origin": "Original.",
    "tags": ["accounts-receivable", "reconciliation", "money"],
    "domain": "finance-economics",
}


def reconcile_invoices(invoices, payments, tolerance_cents):
    """Reconcile payments against invoices in integer cents.

    invoices and payments list [invoice_id, amount_cents]; a payment for an id with no invoice is
    unmatched. Per invoice, with paid = the sum of its payments and diff = paid - amount_cents:
    "unpaid" if paid == 0, else "settled" if abs(diff) <= tolerance_cents, else "short" if
    diff < 0, else "over".
    Returns [labels in invoice order, open_cents = sum of -diff over unpaid and short invoices,
    unmatched_cents].
    Raises ValueError if tolerance_cents < 0, an invoice id repeats, or any amount is below 1.
    """
    if tolerance_cents < 0:
        raise ValueError
    paid, unmatched = {}, 0
    for invoice_id, amount in invoices:
        if invoice_id in paid or amount < 1:
            raise ValueError
        paid[invoice_id] = 0
    for invoice_id, amount in payments:
        if amount < 1:
            raise ValueError
        if invoice_id in paid:
            paid[invoice_id] += amount
        else:
            unmatched += amount
    labels, open_cents = [], 0
    for invoice_id, amount in invoices:
        diff = paid[invoice_id] - amount
        if paid[invoice_id] == 0:
            label = "unpaid"
        elif abs(diff) <= tolerance_cents:
            label = "settled"
        else:
            label = "short" if diff < 0 else "over"
        if label in ("unpaid", "short"):
            open_cents -= diff
        labels.append(label)
    return [labels, open_cents, unmatched]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], [], 0],
        [[["A1", 500]], [], 0],
        [[["A1", 500]], [["A1", 500]], 0],
        [[["A1", 500]], [["A1", 499]], 0],
        [[["A1", 500]], [["A1", 499]], 1],
        [[["A1", 500]], [["A1", 501]], 1],
        [[["A1", 500]], [["A1", 502]], 1],
        [[["A1", 500]], [["A1", 497]], 2],
        [[["A1", 500]], [["A1", 200], ["A1", 300]], 0],
        [[["A1", 500]], [["B2", 300]], 0],
        [[["A1", 500], ["B2", 300]], [["B2", 300], ["C3", 40]], 0],
        [[["A1", 500], ["B2", 300]], [["A1", 450], ["B2", 350]], 5],
        [[["A1", 1]], [["A1", 1]], 0],
        [[["A1", 1], ["A1", 2]], [], 0],
        [[["A1", 0]], [], 0],
        [[["A1", 500]], [["A1", 0]], 0],
        [[["A1", 500]], [], -1],
        [[["A1", 10], ["B2", 20]], [["A1", 5], ["B2", 25]], 4],
        [[["", 500]], [["", 500]], 0],
        [[["", 500], ["", 20]], [], 0],
        [[["", 0]], [], 0],
        [[["A1", 500]], [["", 500]], 0],
    ]
    while len(cases) < n:
        ids = rng.sample(["A1", "B2", "C3", "D4", "E5", "F6"], rng.randint(1, 4))
        invoices = [[i, rng.choice([100, 250, 500, 1000, rng.randint(1, 2000)])] for i in ids]
        payments = []
        for invoice_id, amount in invoices:
            parts = rng.choice([0, 1, 1, 1, 2])
            for _ in range(parts):
                payments.append([invoice_id, max(1, amount // parts + rng.choice([0, 0, -1, 1, -3, 3, -50, 50]))])
        if rng.random() < 0.3:
            payments.append([rng.choice(["X9", "Y8"]), rng.randint(1, 300)])
        rng.shuffle(payments)
        cases.append([invoices, payments, rng.choice([0, 0, 1, 2, 5])])
    return cases[:n]
