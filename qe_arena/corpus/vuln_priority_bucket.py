"""Triage a vulnerability into a remediation time bucket. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/security-compliance/services/security-auditor-reports.ts (MIT)."""

META = {
    "entry": "vuln_priority_bucket",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/domains/security-compliance/services/security-auditor-reports.ts",
    "tags": ["security", "triage"],
}


def vuln_priority_bucket(severity, effort):
    """Return the remediation bucket for a vulnerability.

    severity is case-sensitive; effort is one of "trivial", "minor", "moderate", "major".
    - "critical" -> "immediate"
    - "high" -> "immediate" if effort is "trivial" or "minor", else "short_term"
    - "medium" -> "long_term" if effort is "major", else "medium_term"
    - "low" or "informational" -> "long_term"
    - any other severity -> "accepted"
    """
    if severity == "critical":
        return "immediate"
    if severity == "high":
        return "immediate" if effort == "trivial" or effort == "minor" else "short_term"
    if severity == "medium":
        return "long_term" if effort == "major" else "medium_term"
    if severity == "low" or severity == "informational":
        return "long_term"
    return "accepted"


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    severities = ["critical", "high", "medium", "low", "informational", "unknown", "High"]
    efforts = ["trivial", "minor", "moderate", "major"]
    cases = [[s, e] for s in severities[:5] for e in efforts]
    cases += [["unknown", "major"], ["High", "trivial"], ["", "minor"]]
    while len(cases) < n:
        cases.append([rng.choice(severities), rng.choice(efforts)])
    return cases[:n]
