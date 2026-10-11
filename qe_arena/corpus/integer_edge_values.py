"""Choose integer test values for a bounded schema field. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/data_mocking_agent.py (MIT)."""

META = {
    "entry": "integer_edge_values",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/data_mocking_agent.py",
    "tags": ["test-data", "boundary-values"],
}


def integer_edge_values(minimum, maximum, strategy):
    """Return the integers a test-data generator should try for a bounded field.

    minimum defaults to 0 and maximum to 1000000 when None.
    - strategy "boundary": [minimum, maximum] (both, even when equal).
    - strategy "edge_cases": minimum, minimum + 1, maximum - 1, maximum in that order,
      keeping only values inside [minimum, maximum] and skipping duplicates.
    - any other strategy: [(minimum + maximum) // 2].
    Raises ValueError if minimum > maximum.
    """
    if minimum is None:
        minimum = 0
    if maximum is None:
        maximum = 1000000
    if minimum > maximum:
        raise ValueError("minimum exceeds maximum")
    if strategy == "boundary":
        return [minimum, maximum]
    if strategy == "edge_cases":
        values = []
        for value in (minimum, minimum + 1, maximum - 1, maximum):
            if minimum <= value <= maximum and value not in values:
                values.append(value)
        return values
    return [(minimum + maximum) // 2]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [None, None, "edge_cases"],
        [None, None, "boundary"],
        [None, None, "random"],
        [5, 5, "edge_cases"],
        [5, 5, "boundary"],
        [5, 6, "edge_cases"],
        [5, 7, "edge_cases"],
        [5, 8, "edge_cases"],
        [-3, None, "edge_cases"],
        [None, -1, "edge_cases"],
        [None, 1, "edge_cases"],
        [6, 5, "boundary"],
        [-7, -2, "mid"],
        [-7, -4, "mid"],
        [1, 2, "mid"],
    ]
    while len(cases) < n:
        low = rng.choice([None, rng.randint(-20, 20)])
        high = rng.choice([None, rng.randint(-20, 30)])
        cases.append([low, high, rng.choice(["edge_cases", "boundary", "random"])])
    return cases[:n]
