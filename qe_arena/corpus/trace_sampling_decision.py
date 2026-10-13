"""Decide whether a tracing span is sampled. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/config/enhanced_tracing_config.py (MIT)."""

META = {
    "entry": "trace_sampling_decision",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/config/enhanced_tracing_config.py",
    "tags": ["observability", "sampling"],
    "domain": "software-engineering",
}


def trace_sampling_decision(span_name, trace_id, has_error, base_rate_pct, parent_sampled):
    """Return True if the span should be recorded. Rules are applied in order:

    1. has_error is True -> True.
    2. span_name is exactly "/health", "/metrics" or "health_check" -> False.
    3. span_name contains any critical operation ("generate_tests", "execute_agent",
       "analyze_spec", "authenticate") -> trace_id % 100 < min(base_rate_pct * 5, 100).
    4. parent_sampled is True or False -> that value.
    5. Otherwise (parent_sampled is None) -> trace_id % 100 < base_rate_pct.
    trace_id is a non-negative int and base_rate_pct an int percentage.
    """
    if has_error:
        return True
    if span_name in ("/health", "/metrics", "health_check"):
        return False
    critical = ("generate_tests", "execute_agent", "analyze_spec", "authenticate")
    if any(op in span_name for op in critical):
        return trace_id % 100 < min(base_rate_pct * 5, 100)
    if parent_sampled is not None:
        return parent_sampled
    return trace_id % 100 < base_rate_pct


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["/health", 0, True, 10, False],
        ["/health", 0, False, 10, True],
        ["/metrics", 5, False, 100, None],
        ["health_check", 1, False, 50, None],
        ["/healthz", 1, False, 50, None],
        ["generate_tests", 49, False, 10, False],
        ["generate_tests", 50, False, 10, True],
        ["run:execute_agent", 149, False, 10, None],
        ["analyze_spec", 199, False, 30, None],
        ["authenticate_user", 99, False, 20, None],
        ["authenticate_user", 100, False, 0, True],
        ["list_users", 5, False, 10, True],
        ["list_users", 5, False, 10, False],
        ["list_users", 9, False, 10, None],
        ["list_users", 10, False, 10, None],
        ["list_users", 110, False, 10, None],
        ["list_users", 299, False, 100, None],
    ]
    names = ["generate_tests", "execute_agent", "analyze_spec", "authenticate", "/health",
             "/metrics", "list_users", "get_spec", "health_check", "db.query"]
    while len(cases) < n:
        cases.append([rng.choice(names), rng.randint(0, 999), rng.random() < 0.15,
                      rng.randint(0, 30), rng.choice([None, True, False])])
    return cases[:n]
