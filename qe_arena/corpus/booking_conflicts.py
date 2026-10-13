"""Find meeting-room bookings that clash once a cleaning turnover gap is required. Original."""

META = {
    "entry": "booking_conflicts",
    "origin": "Original.",
    "tags": ["scheduling", "intervals", "calendar"],
    "domain": "office-white-collar",
}


def booking_conflicts(bookings, turnover):
    """Return clashing meeting-room bookings as a list of [i, j, shortfall], i < j, sorted.

    bookings are [room, start, end] with integer minutes 0 <= start < end <= 1440; a room
    needs turnover (int >= 0) free minutes between meetings. i and j clash when the room is
    the same and start_i < end_j + turnover and start_j < end_i + turnover; shortfall is
    min(end_i, end_j) + turnover - max(start_i, start_j). Raises ValueError for a negative
    turnover or a booking time outside 0 <= start < end <= 1440.
    """
    if turnover < 0:
        raise ValueError("turnover must be >= 0")
    for room, start, end in bookings:
        if not 0 <= start < end <= 1440:
            raise ValueError("bad booking time")
    clashes = []
    for i in range(len(bookings)):
        room_i, start_i, end_i = bookings[i]
        for j in range(i + 1, len(bookings)):
            room_j, start_j, end_j = bookings[j]
            if room_i == room_j and start_i < end_j + turnover and start_j < end_i + turnover:
                clashes.append([i, j, min(end_i, end_j) + turnover - max(start_i, start_j)])
    return clashes


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], 0],
        [[["Oak", 540, 600], ["Oak", 600, 660]], 0],
        [[["Oak", 540, 600], ["Oak", 600, 660]], 10],
        [[["Oak", 540, 600], ["Oak", 610, 660]], 10],
        [[["Oak", 540, 600], ["Oak", 609, 660]], 10],
        [[["Oak", 540, 600], ["Elm", 570, 630]], 0],
        [[["Oak", 0, 30], ["Oak", 1380, 1440], ["Oak", 20, 40]], 0],
        [[["Oak", 600, 660], ["Oak", 540, 600], ["Oak", 630, 700]], 5],
        [[["Oak", 600, 660], ["Oak", 540, 600]], 0],
        [[["Oak", 540, 600]], -1],
        [[["Oak", 600, 600]], 0],
        [[["Oak", -1, 30]], 0],
        [[["Oak", 1400, 1441]], 0],
        [[["Elm", 480, 540], ["Oak", 700, 650]], 0],
    ]
    rooms = ["A", "B", "C"]
    while len(cases) < n:
        bookings = []
        for _ in range(rng.choice([2, 2, 3, 4])):
            start = 5 * rng.randint(96, 216)
            bookings.append([rng.choice(rooms), start, start + 5 * rng.randint(3, 24)])
        cases.append([bookings, rng.choice([0, 0, 5, 10, 15])])
    return cases[:n]
