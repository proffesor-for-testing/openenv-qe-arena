"""List the invalid string lengths for a negative boundary test. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/functional_negative_agent.py (MIT)."""

META = {
    "entry": "string_length_boundaries",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/functional_negative_agent.py",
    "tags": ["negative-testing", "boundary-values"],
}


def string_length_boundaries(min_length, max_length):
    """Return the string lengths a negative test should send for a constrained parameter.

    Either bound may be None (unconstrained). If min_length is not None and > 0, add 0
    and then min_length - 1 (min_length - 1 is not added again when it equals 0). If
    max_length is not None, add max_length + 1 and max_length + 100. Lengths are
    returned in that order. Raises ValueError if a given bound is negative, or if both
    are given and min_length > max_length.
    """
    for bound in (min_length, max_length):
        if bound is not None and bound < 0:
            raise ValueError("bounds must be non-negative")
    if min_length is not None and max_length is not None and min_length > max_length:
        raise ValueError("min_length exceeds max_length")
    lengths = []
    if min_length is not None and min_length > 0:
        lengths.append(0)
        if min_length - 1 > 0:
            lengths.append(min_length - 1)
    if max_length is not None:
        lengths.append(max_length + 1)
        lengths.append(max_length + 100)
    return lengths


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [None, None],
        [0, None],
        [1, None],
        [2, None],
        [None, 0],
        [None, 5],
        [3, 3],
        [3, 2],
        [1, 1],
        [0, 0],
        [-1, None],
        [None, -1],
        [5, 40],
    ]
    while len(cases) < n:
        low = rng.choice([None, rng.randint(0, 10)])
        high = rng.choice([None, rng.randint(0, 20)])
        cases.append([low, high])
    return cases[:n]
