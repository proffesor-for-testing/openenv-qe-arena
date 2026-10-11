"""Check load-test metrics against configured performance thresholds. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/performance_agent.py (MIT)."""

META = {
    "entry": "threshold_compliance",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/orchestration_service/agents/performance_agent.py",
    "tags": ["performance", "quality-gate"],
}


def threshold_compliance(metrics, thresholds):
    """Check performance metrics against whichever thresholds are configured.

    metrics has integer keys: avg_ms, p95_ms, throughput_rps, success_pct,
    failed_requests, total_requests. thresholds may contain any of
    "avg_response_time_ms", "p95_response_time_ms", "throughput_rps",
    "success_rate_percent", "error_rate_percent"; absent keys are not checked and do
    not appear in the result. Result keys: "avg_response_time", "p95_response_time",
    "throughput", "success_rate", "error_rate" mapped to bool. Latencies and the error
    rate pass when <= their threshold; throughput and success rate pass when >= theirs.
    error rate = failed_requests * 100 / total_requests, or 0 when total_requests is 0.
    """
    compliance = {}
    if "avg_response_time_ms" in thresholds:
        compliance["avg_response_time"] = metrics["avg_ms"] <= thresholds["avg_response_time_ms"]
    if "p95_response_time_ms" in thresholds:
        compliance["p95_response_time"] = metrics["p95_ms"] <= thresholds["p95_response_time_ms"]
    if "throughput_rps" in thresholds:
        compliance["throughput"] = metrics["throughput_rps"] >= thresholds["throughput_rps"]
    if "success_rate_percent" in thresholds:
        compliance["success_rate"] = metrics["success_pct"] >= thresholds["success_rate_percent"]
    if "error_rate_percent" in thresholds:
        total = metrics["total_requests"]
        error_rate = metrics["failed_requests"] * 100 / total if total > 0 else 0
        compliance["error_rate"] = error_rate <= thresholds["error_rate_percent"]
    return compliance


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def _metrics(avg, p95, rps, success, failed, total):
        return {"avg_ms": avg, "p95_ms": p95, "throughput_rps": rps,
                "success_pct": success, "failed_requests": failed, "total_requests": total}

    all_keys = {"avg_response_time_ms": 200, "p95_response_time_ms": 500, "throughput_rps": 50,
                "success_rate_percent": 99, "error_rate_percent": 1}
    cases = [
        [_metrics(100, 300, 60, 100, 0, 100), {}],
        [_metrics(200, 500, 50, 99, 1, 100), all_keys],
        [_metrics(201, 501, 49, 98, 2, 100), all_keys],
        [_metrics(199, 499, 51, 100, 0, 100), all_keys],
        [_metrics(0, 0, 0, 0, 0, 0), {"error_rate_percent": 0}],
        [_metrics(0, 0, 0, 0, 1, 50), {"error_rate_percent": 2}],
        [_metrics(0, 0, 0, 0, 3, 200), {"error_rate_percent": 1}],
        [_metrics(0, 0, 0, 0, 1, 200), {"error_rate_percent": 0}],
        [_metrics(0, 0, 0, 0, 1, 1), {"error_rate_percent": 50}],
        [_metrics(150, 900, 10, 95, 5, 100), {"avg_response_time_ms": 150}],
        [_metrics(150, 900, 10, 95, 5, 100), {"p95_response_time_ms": 900, "throughput_rps": 11}],
        [_metrics(150, 900, 10, 95, 5, 100), {"success_rate_percent": 95, "error_rate_percent": 5}],
        [_metrics(150, 900, 10, 95, 5, 100), {"success_rate_percent": 96, "error_rate_percent": 4}],
    ]
    keys = list(all_keys)
    while len(cases) < n:
        total = rng.randint(0, 200)
        failed = rng.randint(0, total) if total else 0
        metrics = _metrics(rng.randint(50, 400), rng.randint(100, 900), rng.randint(0, 100),
                           rng.randint(80, 100), failed, total)
        chosen = rng.sample(keys, rng.randint(1, 5))
        bounds = {"avg_response_time_ms": (50, 400), "p95_response_time_ms": (100, 900),
                  "throughput_rps": (0, 100), "success_rate_percent": (80, 100), "error_rate_percent": (0, 20)}
        cases.append([metrics, {k: rng.randint(*bounds[k]) for k in chosen}])
    return cases[:n]
