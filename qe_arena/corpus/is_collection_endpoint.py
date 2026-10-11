"""Classify an API endpoint as a list/search collection endpoint. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/functional_negative_agent.py (MIT)."""

META = {
    "entry": "is_collection_endpoint",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/functional_negative_agent.py",
    "tags": ["api", "classification", "strings"],
}


def is_collection_endpoint(path, method, query_param_count):
    """Return True if an endpoint looks like a collection (list/search) endpoint.

    path and method are compared case-insensitively. True if any of these holds:
    - the method is GET and the path contains none of "{id}", "{uuid}", "/{", "/:";
    - the path contains "search";
    - the path contains "filter";
    - query_param_count is greater than 2.
    Otherwise False.
    """
    path = path.lower()
    method = method.lower()
    indicators = [
        method == "get" and not any(marker in path for marker in ("{id}", "{uuid}", "/{", "/:")),
        "search" in path,
        "filter" in path,
        query_param_count > 2,
    ]
    return any(indicators)


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["/users", "GET", 0],
        ["/users", "get", 0],
        ["/users/{id}", "GET", 0],
        ["/users/{userId}", "GET", 0],
        ["/users/:id", "GET", 0],
        ["/x{uuid}", "GET", 0],
        ["/x{id}", "GET", 0],
        ["/users", "POST", 0],
        ["/users", "POST", 2],
        ["/users", "POST", 3],
        ["/users/{id}", "GET", 3],
        ["/users/{id}/Search", "POST", 0],
        ["/orders/{id}/FILTER", "DELETE", 1],
        ["/orders/{id}/research", "PUT", 0],
        ["/filters/{id}", "PATCH", 2],
        ["/items/{id}", "patch", 2],
    ]
    segments = ["users", "orders", "search", "filter", "{id}", "{uuid}", ":id", "{slug}", "items", "Report"]
    while len(cases) < n:
        path = "/" + "/".join(rng.choice(segments) for _ in range(rng.randint(1, 3)))
        cases.append([path, rng.choice(["GET", "get", "POST", "PUT", "DELETE"]), rng.randint(0, 4)])
    return cases[:n]
