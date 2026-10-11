"""Truncate a prompt to fit a token budget with a binary search. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/llm_providers/utils/token_counter.py (MIT)."""

META = {
    "entry": "truncate_to_token_budget",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/llm_providers/utils/token_counter.py",
    "tags": ["llm", "strings", "binary-search"],
}


def estimate_tokens(text):
    """Approximate token count: one token per 4 characters, rounded up."""
    return (len(text) + 3) // 4


def truncate_to_token_budget(text, max_tokens):
    """Return text truncated so that its estimated token count fits max_tokens.

    estimate_tokens(s) is ceil(len(s) / 4). If estimate_tokens(text) <= max_tokens the
    text is returned unchanged. Otherwise the longest prefix p with
    estimate_tokens(p) <= max_tokens is found and p + "..." is returned. If that prefix
    is empty (max_tokens == 0), the result is text[:10] + "...".
    Raises ValueError if max_tokens is negative.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if estimate_tokens(text) <= max_tokens:
        return text
    left, right = 0, len(text)
    result = ""
    while left < right:
        mid = (left + right + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            result = text[:mid]
            left = mid
        else:
            right = mid - 1
    return result + "..." if result else text[:10] + "..."


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    sample = "abcdefghijklmnopqrstuvwxyz0123456789ABCD"
    cases = [
        ["", 0],
        ["", 3],
        ["abc", 0],
        ["abcd", 1],
        ["abcde", 1],
        ["abcdefgh", 1],
        ["abcdefgh", 2],
        ["abcdefghi", 2],
        [sample[:12], 2],
        [sample[:13], 3],
        [sample[:16], 3],
        [sample[:17], 4],
        [sample, 9],
        [sample, 10],
        [sample, 1],
        [sample[:11], 0],
        ["abc", -1],
    ]
    while len(cases) < n:
        length = rng.randint(0, 40)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        cases.append([text, rng.randint(0, 11)])
    return cases[:n]
