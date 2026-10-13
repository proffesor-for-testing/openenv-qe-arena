"""Validate one CSV row against a column schema and list the problems. Original."""

META = {
    "entry": "validate_csv_row",
    "origin": "Original.",
    "tags": ["csv", "validation", "data-entry"],
    "domain": "office-white-collar",
}


def validate_csv_row(row, schema):
    """Check one CSV row (a list of cell strings) against a schema; return the problems found.

    schema is a list of columns {"name": str, "kind": "int" or "text", "required": bool,
    "min": int, "max": int}. If len(row) != len(schema) the result is exactly
    ["expected <len(schema)> columns, got <len(row)>"]. Otherwise each cell is first stripped
    of leading and trailing spaces (" " only) and checked against its column, in order, giving
    at most one problem per column:
    - blank cell: "<name>: required" if the column is required, else no problem;
    - "int": an optional leading "-" then one or more ASCII digits 0-9, else
      "<name>: not an integer"; an integer outside min..max inclusive is "<name>: out of range";
    - "text": a length outside min..max inclusive is "<name>: bad length".
    Returns [] for a valid row. Raises ValueError when a column's kind is neither "int" nor
    "text" (even if its cell is blank).
    """
    if len(row) != len(schema):
        return ["expected " + str(len(schema)) + " columns, got " + str(len(row))]
    problems = []
    for cell, column in zip(row, schema):
        name, kind, value = column["name"], column["kind"], cell.strip(" ")
        if kind != "int" and kind != "text":
            raise ValueError("unknown column kind: " + str(kind))
        if not value:
            if column["required"]:
                problems.append(name + ": required")
        elif kind == "text":
            if not column["min"] <= len(value) <= column["max"]:
                problems.append(name + ": bad length")
        else:
            digits = value[1:] if value[:1] == "-" else value
            if not digits or any(c not in "0123456789" for c in digits):
                problems.append(name + ": not an integer")
            elif not column["min"] <= int(value) <= column["max"]:
                problems.append(name + ": out of range")
    return problems


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def _column(name, kind, required, low, high):
        return {"name": name, "kind": kind, "required": required, "min": low, "max": high}

    invoice = [_column("invoice", "text", True, 3, 8), _column("qty", "int", True, 1, 500),
               _column("discount", "int", False, -50, 0), _column("note", "text", False, 0, 12)]
    cases = [
        [["INV-1", "12", "-5", "rush"], invoice],
        [["INV", "1", "-50", ""], invoice],
        [["INVOICE1", "500", "0", "twelve chars"], invoice],
        [["IN", "0", "-51", "thirteen char"], invoice],
        [["INVOICE12", "501", "1", ""], invoice],
        [["  INV-2 ", " 7 ", "  ", "   "], invoice],
        [["", "", "", ""], invoice],
        [["INV-3", "-", "5-", "x"], invoice],
        [["INV-4", "1.5", "--2", "x"], invoice],
        [["INV-5", "+3", "-0", "x"], invoice],
        [["INV-6", "3"], invoice],
        [[], []],
        [["a"], [_column("a", "date", False, 0, 0)]],
        [[" "], [_column("a", "money", False, 0, 0)]],
    ]
    pieces = ["", " ", "-", "0", "7", "12", "250", "499", "500", "501", "1x", "ab", "INV-77", "-49", "-50", "-51"]
    while len(cases) < n:
        row = ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 3))) for _ in invoice]
        if rng.random() < 0.08:
            row = row[:rng.randint(0, 3)]
        cases.append([row, invoice])
    return cases[:n]
