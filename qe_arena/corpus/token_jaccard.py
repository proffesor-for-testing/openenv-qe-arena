"""Jaccard similarity of two texts over normalized word tokens. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/learning/qe-flywheel/corpus-scorer.ts (MIT)."""

import re

META = {
    "entry": "token_jaccard",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/learning/qe-flywheel/corpus-scorer.ts",
    "tags": ["similarity", "strings", "sets"],
    "domain": "software-engineering",
}


def tokenize(text):
    """Lowercase text and return the set of alphanumeric tokens of length >= 2."""
    return {tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if len(tok) >= 2}


def token_jaccard(a, b):
    """Return |A & B| / |A | B| over the token sets of a and b, rounded to 4 decimals.

    Tokens are maximal runs of [a-z0-9] after lowercasing; tokens shorter than 2
    characters are dropped. If both token sets are empty the result is 0.0.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta) + len(tb) - inter
    return round(inter / union, 4)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["", ""],
        ["a b", "c"],
        ["login", ""],
        ["login", "LOGIN"],
        ["login page", "login form"],
        ["a login", "b login"],
        ["test_user-42", "TEST user 42"],
        ["x y z", "x"],
        ["api timeout retry", "retry api timeout"],
        ["flaky test in ci", "ci flaky"],
        ["ab", "ab ab ab"],
    ]
    words = ["login", "api", "Timeout", "retry", "ci", "x", "test", "flaky", "db-01", "Error", "a"]
    while len(cases) < n:
        cases.append([" ".join(rng.choice(words) for _ in range(rng.randint(0, 5))),
                      " ".join(rng.choice(words) for _ in range(rng.randint(0, 5)))])
    return cases[:n]
