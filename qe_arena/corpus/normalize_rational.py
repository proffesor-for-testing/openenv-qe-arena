"""Parse a rational literal and reduce it to lowest terms. Original."""

import math
import re

META = {
    "entry": "normalize_rational",
    "origin": "Original.",
    "tags": ["number-theory", "parsing", "rationals"],
    "domain": "mathematics-or-formal-reasoning",
}


def normalize_rational(text):
    """Parse a rational literal and return it in lowest terms as [numerator, denominator].

    Surrounding whitespace is ignored. Accepted forms, where a and b are ASCII decimal integers
    with an optional "+" or "-" and n, d are unsigned ASCII decimal integers:
    "a" (an integer), "a/b" (a fraction; b may be negative) and "a n/d" (a mixed number, one
    space between a and n/d, meaning a whole part plus a proper fraction). A mixed number needs
    n < d and takes the sign written on a, so "-1 1/2" is -3/2 and "-0 1/2" is -1/2.
    The result has a positive denominator and gcd(|numerator|, denominator) == 1; zero is [0, 1].
    Raises ValueError for any other text, a zero denominator, or a mixed number with n >= d.
    """
    found = re.fullmatch(r"([+-]?[0-9]+)(?: ([0-9]+)/([0-9]+)|/([+-]?[0-9]+))?", text.strip())
    if found is None:
        raise ValueError("invalid rational literal: " + text)
    whole, part, part_den, den_text = found.groups()
    if part is not None:
        num, den = int(part), int(part_den)
        if num >= den:
            raise ValueError("invalid rational literal: " + text)
        num = abs(int(whole)) * den + num
        if whole.startswith("-"):
            num = -num
    else:
        num, den = int(whole), int(den_text or "1")
    if den < 0:
        num, den = -num, -den
    if den < 1:
        raise ValueError("invalid rational literal: " + text)
    g = math.gcd(num, den)
    return [num // g, den // g]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["6/8"], ["-6/8"], ["6/-8"], ["-6/-8"], ["+6/8"], ["0/5"], ["0"], ["-0"], ["7"], ["-7"],
        ["5/0"], ["1 1/2"], ["-1 1/2"], ["-0 1/2"], ["+2 3/4"], ["2 4/8"], ["2 4/4"], ["2 5/4"],
        ["2 0/4"], ["2 1/0"], ["2 -1/2"], ["  3/9  "], ["3 / 9"], ["1.5"], [""], ["a/b"],
        ["10/4"], ["5/-1"], ["-5/-1"], ["0/-3"], ["12/-18"], ["0 0/7"], ["3  1/2"], ["-9 2/3"], ["100/25"],
    ]
    while len(cases) < n:
        num, den = rng.randint(-60, 60), rng.choice([rng.randint(-24, 24), rng.randint(1, 24)])
        if rng.random() < 0.3:
            cases.append([f"{rng.randint(-5, 5)} {rng.randint(0, 12)}/{rng.randint(0, 12)}"])
        else:
            cases.append([f"{num}/{den}"])
    return cases[:n]
