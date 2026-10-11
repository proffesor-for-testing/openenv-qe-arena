"""Bucket an API's performance-test planning complexity. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/performance_planner_agent.py (MIT)."""

META = {
    "entry": "api_complexity_level",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/performance_planner_agent.py",
    "tags": ["performance", "planning", "thresholds"],
}


def api_complexity_level(total_endpoints, critical_paths, data_intensive_ops):
    """Return "low", "medium" or "high" for the planning complexity of an API.

    score = total_endpoints + 2 * critical_paths + 3 * data_intensive_ops.
    score < 10 is "low", 10 <= score < 25 is "medium", score >= 25 is "high".
    Raises ValueError if any count is negative, or if critical_paths or
    data_intensive_ops is greater than total_endpoints.
    """
    if min(total_endpoints, critical_paths, data_intensive_ops) < 0:
        raise ValueError("counts must be non-negative")
    if critical_paths > total_endpoints or data_intensive_ops > total_endpoints:
        raise ValueError("subset counts cannot exceed total_endpoints")
    score = total_endpoints + critical_paths * 2 + data_intensive_ops * 3
    if score < 10:
        return "low"
    if score < 25:
        return "medium"
    return "high"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [0, 0, 0],
        [9, 0, 0],
        [10, 0, 0],
        [7, 1, 0],
        [6, 0, 1],
        [8, 1, 0],
        [24, 0, 0],
        [25, 0, 0],
        [18, 2, 1],
        [17, 2, 1],
        [12, 3, 2],
        [3, 3, 3],
        [4, 3, 3],
        [-1, 0, 0],
        [5, -1, 0],
        [5, 0, -2],
        [2, 3, 0],
        [2, 0, 3],
        [3, 3, 0],
    ]
    while len(cases) < n:
        total = rng.randint(0, 20)
        cases.append([total, rng.randint(0, total), rng.randint(0, total)])
    return cases[:n]
