"""Compute SVG badge geometry from label and message lengths. Adapted from proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/badges/generator.py (MIT)."""

META = {
    "entry": "badge_dimensions",
    "origin": "proffesor-for-testing/lionagi-qe-fleet@ce1d58750a240a16c3c53abcb8a7c73864e98c99:src/lionagi_qe/badges/generator.py",
    "tags": ["reporting", "arithmetic"],
}


def badge_dimensions(label, message):
    """Return the integer layout of a two-part status badge (shields.io style).

    With char_width = 6 pixels:
      label_width   = len(label) * 6 + 10
      message_width = len(message) * 6 + 10
      width         = label_width + message_width
      label_x       = label_width * 10 // 2
      message_x     = label_width * 10 + message_width * 10 // 2
      label_text_length   = len(label) * 6 * 10
      message_text_length = len(message) * 6 * 10
    Returns a dict with exactly those seven keys.
    """
    char_width = 6
    label_width = len(label) * char_width + 10
    message_width = len(message) * char_width + 10
    return {
        "width": label_width + message_width,
        "label_width": label_width,
        "message_width": message_width,
        "label_x": (label_width * 10) // 2,
        "message_x": (label_width * 10) + (message_width * 10) // 2,
        "label_text_length": len(label) * char_width * 10,
        "message_text_length": len(message) * char_width * 10,
    }


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["", ""],
        ["a", ""],
        ["", "a"],
        ["coverage", "87%"],
        ["build", "passing"],
        ["tests", "1203 passed"],
        ["quality gate", "failed"],
    ]
    words = ["coverage", "tests", "build", "flaky", "mutation", "92%", "ok", "failed", "A+", "v3.1.0"]
    while len(cases) < n:
        cases.append([rng.choice(words)[: rng.randint(0, 8)], rng.choice(words)])
    return cases[:n]
