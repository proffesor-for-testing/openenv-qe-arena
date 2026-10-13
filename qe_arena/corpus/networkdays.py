"""Count working days between two dates, inclusive, excluding weekends and holidays. Original."""

META = {
    "entry": "networkdays",
    "origin": "Original.",
    "tags": ["calendar", "working-days", "spreadsheet"],
    "domain": "office-white-collar",
}


def weekday(day):
    """Day of the week for a day number counted from 1970-01-01 (a Thursday): Monday 0 .. Sunday 6."""
    return (day + 3) % 7


def networkdays(start, end, holidays):
    """Count working days from start to end, both inclusive, like a spreadsheet NETWORKDAYS.

    Dates are integer day numbers counted from 1970-01-01 = 0 (may be negative). Monday to
    Friday are working days; Saturday and Sunday are not. holidays is a list of day numbers:
    each distinct one that falls on a working day within the range is subtracted once;
    weekend holidays, holidays outside the range and duplicates change nothing.
    When start is after end the count runs from end to start and the result is negated, so
    networkdays(a, b, h) == -networkdays(b, a, h) whenever a != b.
    Raises TypeError if start, end or any holiday is not an int (bools are rejected).
    """
    for value in [start, end] + list(holidays):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("dates must be int day numbers")
    sign = 1
    if start > end:
        start, end, sign = end, start, -1
    weeks, extra = divmod(end - start + 1, 7)
    count = weeks * 5
    first = weekday(start)
    for offset in range(extra):
        if (first + offset) % 7 < 5:
            count += 1
    off = {h for h in holidays if start <= h <= end and weekday(h) < 5}
    return sign * (count - len(off))


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    # 20108 is Monday 2025-01-20; 20112 Friday, 20113 Saturday, 20114 Sunday.
    cases = [
        [20108, 20112, []],
        [20108, 20108, []],
        [20113, 20114, []],
        [20112, 20115, [20113]],
        [20108, 20121, [20108, 20121]],
        [20109, 20119, [20109, 20109, 20113, 20199]],
        [20121, 20108, [20112]],
        [20115, 20112, []],
        [20114, 20114, [20114]],
        [20111, 20111, [20111]],
        [-3, 3, [0, -1]],
        [20108, 20472, [20108, 20261, 20472]],
        [20108, 20112, [True]],
        ["2025-01-20", 20112, []],
    ]
    while len(cases) < n:
        start = rng.randint(19700, 20500)
        end = start + rng.randint(-20, 120)
        lo, hi = min(start, end), max(start, end)
        holidays = [rng.randint(lo - 5, hi + 5) for _ in range(rng.randint(0, 5))]
        cases.append([start, end, holidays])
    return cases[:n]
