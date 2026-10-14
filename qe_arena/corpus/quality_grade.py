"""Grade a quality score, failing on any critical check failure. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/tools/quality-assessment/evaluate.ts (MIT)."""

META = {
    "entry": "quality_grade",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/tools/quality-assessment/evaluate.ts",
    "tags": ["quality-gate", "thresholds"],
    "domain": "software-engineering",
}


def quality_grade(score, checks):
    """Return a letter grade for a quality assessment.

    checks is a list of {"passed": bool, "severity": str}. If any check has
    passed False and severity "critical", the grade is "F" regardless of score.
    Otherwise: score >= 90 "A", >= 80 "B", >= 70 "C", >= 60 "D", else "F".
    Raises ValueError if score is outside 0..100.
    """
    if score < 0 or score > 100:
        raise ValueError("score must be within 0..100")
    if any(not c["passed"] and c["severity"] == "critical" for c in checks):
        return "F"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def check(passed, severity):
        return {"passed": passed, "severity": severity}

    cases = [
        [100, []],
        [90, []],
        [89, []],
        [80, []],
        [79, []],
        [70, []],
        [69, []],
        [60, []],
        [59, []],
        [0, []],
        [95, [check(False, "critical")]],
        [95, [check(True, "critical")]],
        [95, [check(False, "high"), check(True, "critical")]],
        [85, [check(True, "low"), check(False, "critical")]],
        [101, []],
        [-1, []],
    ]
    while len(cases) < n:
        checks = [check(rng.random() < 0.7, rng.choice(["low", "high", "critical"]))
                  for _ in range(rng.randint(0, 3))]
        cases.append([rng.randint(50, 100), checks])
    return cases[:n]
