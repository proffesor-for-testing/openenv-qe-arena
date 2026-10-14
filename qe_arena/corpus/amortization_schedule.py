"""Summarise a level-payment loan in integer cents. Original."""

import math
from fractions import Fraction

META = {
    "entry": "amortization_schedule",
    "origin": "Original.",
    "tags": ["loan", "amortization", "money"],
    "domain": "finance-economics",
}


def amortization_schedule(principal_cents, annual_rate_bp, months):
    """Summarise a fixed-rate level-payment loan in integer cents.

    r = annual_rate_bp / 120000 per month, exact. payment = ceil(principal_cents / months) if the
    rate is 0, else ceil(principal_cents * r / (1 - (1 + r) ** -months)). From balance =
    principal_cents, each month's interest is balance * r rounded half up to a cent. The loan closes
    in the first month where balance + interest <= payment, or in month `months`, by paying
    balance + interest; otherwise balance becomes balance + interest - payment.
    Returns [payment, final_payment, total_interest, months_paid].
    Raises ValueError if principal_cents < 1, months < 1 or annual_rate_bp is outside 0..5000.
    """
    if principal_cents < 1 or months < 1 or not 0 <= annual_rate_bp <= 5000:
        raise ValueError
    if annual_rate_bp == 0:
        payment = math.ceil(Fraction(principal_cents, months))
    else:
        rate = Fraction(annual_rate_bp, 120000)
        payment = math.ceil(principal_cents * rate / (1 - (1 + rate) ** -months))
    balance, total_interest = principal_cents, 0
    for month in range(1, months + 1):
        interest = (balance * annual_rate_bp + 60000) // 120000
        total_interest += interest
        if balance + interest <= payment or month == months:
            return [payment, balance + interest, total_interest, month]
        balance += interest - payment


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [100, 0, 3],
        [10, 0, 7],
        [9, 0, 7],
        [7, 1200, 2],
        [1000, 3000, 3],
        [705, 2132, 7],
        [1229, 2598, 5],
        [1774, 608, 6],
        [59999, 1, 1],
        [100000, 0, 12],
        [100000, 600, 12],
        [100000, 600, 1],
        [1, 600, 1],
        [1, 1, 1],
        [25000000, 450, 360],
        [2000000, 5000, 24],
        [2000000, 4999, 24],
        [500000, 1200, 6],
        [0, 600, 12],
        [100000, 600, 0],
        [100000, -1, 12],
        [100000, 5001, 12],
    ]
    while len(cases) < n:
        months = rng.choice([1, 2, 3, 6, 12, 24, 36, 48, 60, rng.randint(1, 120), rng.choice([180, 240, 360])])
        rate = rng.choice([0, rng.randint(1, 3000), rng.randint(1, 3000), rng.randint(1, 5000)])
        cases.append([rng.randint(1, 50) * rng.choice([1, 100, 1000, 10000]) + rng.randint(0, 99), rate, months])
    return cases[:n]
