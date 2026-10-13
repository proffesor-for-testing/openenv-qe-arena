"""Find meeting-room bookings that clash once a cleaning turnover gap is required. Original."""

META = {
    "entry": "booking_conflicts",
    "origin": "Original.",
    "tags": ["scheduling", "intervals", "calendar"],
    "domain": "office-white-collar",
}


def booking_conflicts(bookings, turnover):
    """Return the index pairs of bookings that clash, as a sorted list of [i, j] with i < j.

    bookings is a list of [room, start, end]: room is a str, start and end are integer minutes
    since midnight with 0 <= start < end <= 1440. turnover (int, >= 0) is the number of free
    minutes a room needs between two meetings. Bookings i and j clash when they name the same
    room and start_i < end_j + turnover and start_j < end_i + turnover. So with turnover 0,
    back-to-back meetings (one ends at 600, the next starts at 600) do not clash; with
    turnover 10 the next one must start at 610 or later.
    Raises ValueError if turnover is negative or any booking violates 0 <= start < end <= 1440
    (checked before any pair is compared).
    """
    if turnover < 0:
        raise ValueError("turnover must be >= 0")
    for room, start, end in bookings:
        if not 0 <= start < end <= 1440:
            raise ValueError("bad booking time for room " + room)
    clashes = []
    for i in range(len(bookings)):
        room_i, start_i, end_i = bookings[i]
        for j in range(i + 1, len(bookings)):
            room_j, start_j, end_j = bookings[j]
            if room_i == room_j and start_i < end_j + turnover and start_j < end_i + turnover:
                clashes.append([i, j])
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
        [[["Oak", 540, 600]], -1],
        [[["Oak", 600, 600]], 0],
        [[["Oak", -1, 30]], 0],
        [[["Oak", 1400, 1441]], 0],
        [[["Elm", 480, 540], ["Oak", 700, 650]], 0],
    ]
    rooms = ["Oak", "Elm", "Ash"]
    while len(cases) < n:
        bookings = []
        for _ in range(rng.randint(2, 7)):
            start = 5 * rng.randint(96, 216)
            bookings.append([rng.choice(rooms), start, start + 5 * rng.randint(3, 24)])
        cases.append([bookings, rng.choice([0, 0, 5, 10, 15])])
    return cases[:n]
