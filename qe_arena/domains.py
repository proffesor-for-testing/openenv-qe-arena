"""Arena domains and the task-ID slugs that scope a task cell to one of them.

Dependency-free so tooling outside the image (scripts/qe_flywheel) can read it without the sandbox.

Every corpus entry names its domain in ``META["domain"]``, spelled as the arena leaderboard spells it.
A task ID without a slug (``qe_mutant_kill-d2``) is the software-engineering cell, exactly as before
domains existed; ``qe_mutant_kill-finance-d2`` draws only finance-economics entries.
"""

from __future__ import annotations

SOFTWARE = "software-engineering"
DOMAINS = (SOFTWARE, "industrial-physical-systems", "natural-science", "office-white-collar",
           "finance-economics", "mathematics-or-formal-reasoning", "cybersecurity", "media-content-production")
# Task-ID slug per domain. Software engineering has none: its cells keep their original IDs.
SLUGS = {
    "industrial-physical-systems": "industrial",
    "natural-science": "science",
    "office-white-collar": "office",
    "finance-economics": "finance",
    "mathematics-or-formal-reasoning": "math",
    "cybersecurity": "security",
    "media-content-production": "media",
}
BY_SLUG = {slug: domain for domain, slug in SLUGS.items()}
