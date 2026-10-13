"""Parse a webhook signature header like "t=123,v1=abc". Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/adapters/a2a/notifications/signature.ts (MIT)."""

META = {
    "entry": "parse_signature_header",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/adapters/a2a/notifications/signature.ts",
    "tags": ["parsing", "security", "webhooks"],
    "domain": "software-engineering",
}


def parse_signature_header(header):
    """Parse "t=<timestamp>,v1=<signature>" into {"timestamp": int, "v1": str}, or None.

    The header is split on ","; in each part the key is the text before the first "="
    and the value is everything after it ("" when there is no "="). Keys other than
    "t" and "v1" are ignored; if a key repeats, the later value wins. Every "t" value
    must be a non-empty string of decimal digits, otherwise the result is None (even
    if a later "t" is valid). Returns None if the header is empty, "t" is missing, or
    "v1" is missing or empty.
    """
    if not header:
        return None
    timestamp = None
    v1 = None
    for part in header.split(","):
        key, _, value = part.partition("=")
        if key == "t":
            if not value.isdecimal():
                return None
            timestamp = int(value)
        elif key == "v1":
            v1 = value
    if timestamp is None or not v1:
        return None
    return {"timestamp": timestamp, "v1": v1}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [""],
        ["t=1"],
        ["v1=abc"],
        ["t=1,v1=abc"],
        ["v1=abc,t=0"],
        ["t=12,v1="],
        ["t=,v1=abc"],
        ["t=1a,v1=abc"],
        ["t,v1=abc"],
        ["t=5,v1=a=b=="],
        ["t=5,v1=x,v1=y"],
        ["t=5,t=6,v1=x"],
        ["t=5,v0=x,v1=z"],
        ["T=5,v1=x"],
        [" t=5,v1=x"],
        ["t=-5,v1=x"],
        ["t=5;v1=x"],
    ]
    while len(cases) < n:
        parts = []
        for _ in range(rng.randint(1, 4)):
            key = rng.choice(["t", "v1", "v0", "x", "t"])
            value = rng.choice([str(rng.randint(0, 9999)), "ab=c", "", "f00d", "9z"])
            parts.append(key + "=" + value if rng.random() < 0.9 else key)
        cases.append([",".join(parts)])
    return cases[:n]
