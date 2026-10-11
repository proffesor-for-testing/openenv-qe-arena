"""Compare two dotted version strings. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/tools/security-compliance/scan.ts (MIT)."""

META = {
    "entry": "compare_versions",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/tools/security-compliance/scan.ts",
    "tags": ["versions", "parsing"],
}


def parse_version(text):
    """Return the first three numeric components of a version string, padded with 0."""
    if text.startswith("v"):
        text = text[1:]
    parts = text.split(".")[:3]
    for part in parts:
        if not part.isdigit():
            raise ValueError("bad version component: " + part)
    numbers = [int(part) for part in parts]
    return numbers + [0] * (3 - len(numbers))


def compare_versions(a, b):
    """Compare versions like "1.2.3": return -1 if a < b, 0 if equal, 1 if a > b.

    One leading "v" is ignored. Only the first three dot-separated components are
    read (later ones are ignored without validation); missing components count as 0,
    so "1.2" equals "1.2.0". Components compare numerically ("1.10" > "1.9").
    Raises ValueError if one of the first three components is empty or not all digits.
    """
    pa, pb = parse_version(a), parse_version(b)
    for i in range(3):
        if pa[i] < pb[i]:
            return -1
        if pa[i] > pb[i]:
            return 1
    return 0


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["1.0.0", "1.0.0"],
        ["1.2", "1.2.0"],
        ["1", "1.0.1"],
        ["1.10.0", "1.9.9"],
        ["1.9.9", "1.10.0"],
        ["2.0.0", "1.99.99"],
        ["0.0.1", "0.0.2"],
        ["0.1.0", "0.0.9"],
        ["v1.2.3", "1.2.3"],
        ["vv1.0", "1.0"],
        ["1.2.3.9", "1.2.3.1"],
        ["1.2.3", "1.2.4.x"],
        ["1..3", "1.0.3"],
        ["1.2.beta", "1.2.0"],
        ["", "0"],
        ["3.0.1", "3.0.0"],
        ["10.0.0", "9.0.0"],
    ]
    while len(cases) < n:
        def version():
            return ".".join(str(rng.randint(0, 12)) for _ in range(rng.randint(1, 4)))
        a = version()
        b = a if rng.random() < 0.2 else version()
        cases.append([rng.choice(["", "v"]) + a, b])
    return cases[:n]
