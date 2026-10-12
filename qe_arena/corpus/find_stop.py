"""Find where generated text is cut at the earliest stop sequence. Ported from Rust; adapted from ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs (MIT)."""

PRE_FIX = "ruvnet/RuVector@3e1d1886c50f2db817664648b95fab18e3e4c34b:crates/ruvllm/src/backends/candle_backend.rs"
META = {
    "entry": "find_stop",
    "origin": "ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs",
    "tags": ["llm", "strings", "streaming"],
    "real_faults": [
        {"id": "find_stop/first-listed", "kind": "first-match-vs-best", "origin": PRE_FIX,
         "summary": "cuts at the first stop sequence in list order that occurs, not at the earliest offset",
         "find": "        if pos != -1 and (earliest is None or pos < earliest):\n            earliest = pos\n",
         "replace": "        if pos != -1:\n            return pos\n"},
        {"id": "find_stop/empty-matches", "kind": "missing-guard", "origin": PRE_FIX,
         "summary": "an empty stop sequence matches at offset 0, so the whole output is cut",
         "find": "        if not stop:\n            continue\n", "replace": ""},
    ],
}


def find_stop(text, stop_sequences):
    """Return the offset in text where the earliest stop sequence starts, or None.

    Every non-empty string in stop_sequences is searched for in text. The result is the
    smallest index at which any of them occurs (0 means the whole text is cut away).
    The order of stop_sequences does not matter. Empty stop sequences never match.
    Returns None when no non-empty stop sequence occurs, including for an empty list.
    """
    earliest = None
    for stop in stop_sequences:
        if not stop:
            continue
        pos = text.find(stop)
        if pos != -1 and (earliest is None or pos < earliest):
            earliest = pos
    return earliest


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["", []],
        ["hello", []],
        ["hello", ["lo", "l"]],
        ["hello", ["", "lo", "l"]],
        ["hey", ["", "lo", "l"]],
        ["abc", [""]],
        ["", ["a"]],
        ["abcabc", ["c", "b"]],
        ["abcabc", ["b", "c"]],
        ["a\nUser: hi", ["User:", "\n"]],
        ["done</s>more", ["</s>", "<im_end>"]],
        ["x<im_end>y</s>", ["</s>", "<im_end>"]],
        ["aaaa", ["aa", "a"]],
    ]
    alphabet = "abcab\n "
    while len(cases) < n:
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 16)))
        stops = []
        for _ in range(rng.randint(0, 3)):
            if rng.random() < 0.12:
                stops.append("")
            else:
                stops.append("".join(rng.choice("abc\n") for _ in range(rng.randint(1, 3))))
        cases.append([text, stops])
    return cases[:n]
