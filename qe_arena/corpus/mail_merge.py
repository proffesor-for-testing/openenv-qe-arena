"""Fill a mail-merge template from one record, applying defaults and reporting missing fields. Original."""

META = {
    "entry": "mail_merge",
    "origin": "Original.",
    "tags": ["templating", "mail-merge", "text"],
    "domain": "office-white-collar",
}


def mail_merge(template, record):
    """Fill the {{field}} placeholders of template from record; return [text, missing].

    Each "{{" opens a placeholder closed by the next "}}", which must come before any later
    "{{"; other text is copied. A body is "field" or "field|default"; the field is stripped
    of spaces and must be 1 to 32 characters. A missing, None or "" value is replaced by the
    default ("" without "|"), and without "|" the field is also missing. Other values go in
    as str(value). missing is sorted and distinct. Raises ValueError for an unclosed
    placeholder or a bad field length.
    """
    pieces = template.split("{{")
    out, missing = [pieces[0]], set()
    for piece in pieces[1:]:
        body, closed, rest = piece.partition("}}")
        if not closed:
            raise ValueError("unclosed placeholder")
        field, bar, default = body.partition("|")
        field = field.strip(" ")
        if len(field) < 1 or len(field) > 32:
            raise ValueError("field name must be 1 to 32 characters")
        value = record.get(field)
        if value is None or value == "":
            if not bar:
                missing.add(field)
            value = default
        out.append(str(value) + rest)
    return ["".join(out), sorted(missing)]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["Dear {{name}},", {"name": "Ana"}],
        ["Dear {{ title |Sir}},", {"title": None}],
        ["Ref {{id}} at {{city|HQ}}", {"id": 42, "city": ""}],
        ["{{city}}{{title}}!", {"city": "", "title": None}],
        ["no fields", {}],
        ["", {"a": 1}],
        ["{{a}}{{a}}}}", {"a": "x"}],
        ["{{ a|}} {{zip}} {{zip}}", {}],
        ["a {b} {{b}", {"b": 2}],
        ["{{ |x}} y", {}],
        ["{{}} {{a}}", {}],
        ["{{ abcdefghijklmnopqrstuvwxyz012345 }}!", {"abcdefghijklmnopqrstuvwxyz012345": "ok"}],
        ["{{abcdefghijklmnopqrstuvwxyz0123456}}", {}],
        ["{{a|{{b}}", {"a": "A"}],
        ["{{a}} {{b|B}} {{c}}", {"a": 0, "b": "", "c": "C"}],
    ]
    words = ["Hi", "Re:", "-", ",", "{", "}"]
    fields = ["name", "zip", "id", " name ", "zip|00", "id|", "city|HQ"]
    records = [{}, {"name": "Ana"}, {"name": "", "id": 7}, {"zip": None, "id": 0}, {"city": "Rome"}]
    while len(cases) < n:
        parts = [rng.choice(words) if rng.random() < 0.4 else "{{" + rng.choice(fields) + "}}"
                 for _ in range(rng.randint(1, 3))]
        cases.append([rng.choice([" ", ""]).join(parts), rng.choice(records)])
    return cases[:n]
