"""Fill a mail-merge template from one record, applying defaults and reporting missing fields. Original."""

META = {
    "entry": "mail_merge",
    "origin": "Original.",
    "tags": ["templating", "mail-merge", "text"],
    "domain": "office-white-collar",
}


def mail_merge(template, record):
    """Fill the {{field}} placeholders of template from record (a dict of str keys).

    Every "{{" opens a placeholder that must be closed by a "}}" before the next "{{" (or the
    end of the template); text after that "}}" is copied unchanged, as is the text before the
    first "{{". A placeholder body is "field" or "field|default"; the field name has
    surrounding spaces removed ("{{ name }}" is field "name") and must be 1 to 32 characters.
    A field whose value is missing, None or "" is blank: it is replaced by its default (""
    when there is no "|"), and a blank field with no "|" is also reported as missing. Other
    values are inserted with str(), so 7 becomes "7".
    Returns {"text": the filled text, "missing": sorted list of distinct missing fields}.
    Raises ValueError for an unclosed placeholder or a field name of 0 or more than 32 characters.
    """
    pieces = template.split("{{")
    out, missing = [pieces[0]], set()
    for piece in pieces[1:]:
        body, closed, rest = piece.partition("}}")
        if not closed:
            raise ValueError("unclosed placeholder: {{" + piece)
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
    return {"text": "".join(out), "missing": sorted(missing)}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    person = {"first": "Ana", "last": "Ilic", "city": "", "id": 42, "title": None}
    cases = [
        ["Dear {{first}} {{last}},", person],
        ["Dear {{ title |Customer}},", person],
        ["Ref {{id}} from {{city|our office}}.", person],
        ["Hi {{city}}{{title}}!", person],
        ["{{first}}", person],
        ["no fields here", person],
        ["", person],
        ["{{first}}{{first}}}}", person],
        ["{{ last|}} and {{zip}} {{zip}}", person],
        ["a {first} {{first}", person],
        ["{{ |x}} y", person],
        ["{{}} {{first}}", {}],
        ["{{ abcdefghijklmnopqrstuvwxyz012345 }}!", {"abcdefghijklmnopqrstuvwxyz012345": "ok"}],
        ["{{abcdefghijklmnopqrstuvwxyz0123456}}", {}],
        ["{{a|{{b}}", {"a": "A"}],
        ["x {{a}} {{b|B}} {{c}}", {"a": 0, "b": "", "c": "C"}],
    ]
    words = ["Dear", "Hello", "Order", "for", "-", "your", ",", "{", "}"]
    fields = ["first", "last", "city", "id", "title", "zip", " first ", "last|n/a", "zip|", "city|HQ", "x"]
    while len(cases) < n:
        parts = []
        for _ in range(rng.randint(1, 6)):
            parts.append(rng.choice(words) if rng.random() < 0.5 else "{{" + rng.choice(fields) + "}}")
        cases.append([" ".join(parts) if rng.random() < 0.7 else "".join(parts), person])
    return cases[:n]
