"""Validate one invoice-line CSV row against its column schema and list the problems. Original."""

META = {
    "entry": "validate_csv_row",
    "origin": "Original.",
    "tags": ["csv", "validation", "data-entry"],
    "domain": "office-white-collar",
}


def validate_csv_row(row):
    """Check a CSV row of str cells [ref, qty, note]; return its problems in column order.

    A row without exactly 3 cells gives ["bad width"]. Each cell is stripped of " " first.
    ref (required text, 3-8 chars), qty (required int, 1-500), note (optional text, 2-12
    chars). Per cell at most one problem: blank required cell: "<name>: required"; qty that
    is not an optional "-" plus digits 0-9: "qty: not int", else outside 1-500: "qty: range";
    text length outside its range: "<name>: length". Raises TypeError for a non-str cell.
    """
    if len(row) != 3:
        return ["bad width"]
    problems = []
    for cell, (name, low, high) in zip(row, [("ref", 3, 8), ("qty", 1, 500), ("note", 2, 12)]):
        if not isinstance(cell, str):
            raise TypeError("cells must be str")
        value = cell.strip(" ")
        if not value:
            if name != "note":
                problems.append(name + ": required")
        elif name != "qty":
            if not low <= len(value) <= high:
                problems.append(name + ": length")
        else:
            digits = value[1:] if value[:1] == "-" else value
            if not digits or any(c not in "0123456789" for c in digits):
                problems.append("qty: not int")
            elif not low <= int(value) <= high:
                problems.append("qty: range")
    return problems


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [["INV-1", "12", "rush"]],
        [["INV", "1", "ok"]],
        [["INVOICE1", "500", "twelve chars"]],
        [["IN", "0", "thirteen char"]],
        [["INVOICE12", "501", "x"]],
        [[" INV-2 ", " 7 ", " "]],
        [["", "", ""]],
        [["INV-3", "-", "ab"]],
        [["INV-4", "1.5", "ab"]],
        [["INV-5", "+3", "ab"]],
        [["INV-6", "-5", "ab"]],
        [["INV-7", "--2", "ab"]],
        [["INV-8", "3"]],
        [["INV-9", 12, ""]],
        [[]],
        [["A", "2", "", "x"]],
    ]
    refs = ["", " ", "AB", "ABC", "INV-77", "INVOICE9", "INVOICE10"]
    qtys = ["", "-", "0", "1", "7", "250", "500", "501", "-3", "1x", " 42 "]
    notes = ["", "x", "ok", "rush", "call first", "twelve chars", "thirteen char"]
    while len(cases) < n:
        row = [rng.choice(refs), rng.choice(qtys), rng.choice(notes)]
        if rng.random() < 0.08:
            row = row[:rng.randint(0, 2)]
        cases.append([row])
    return cases[:n]
