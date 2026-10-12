"""Measure how much generated text is safe to stream before a stop sequence. Ported from Rust; adapted from ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs (MIT)."""

PRE_FIX = "ruvnet/RuVector@3e1d1886c50f2db817664648b95fab18e3e4c34b:crates/ruvllm/src/backends/candle_backend.rs"
META = {
    "entry": "streamable_len",
    "origin": "ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs",
    "tags": ["llm", "streaming", "utf-8"],
    "real_faults": [
        {"id": "streamable_len/no-holdback", "kind": "missing-logic", "origin": PRE_FIX,
         "summary": "streams everything settled, so the start of a stop sequence reaches the client",
         "note": "reconstructed from the pre-fix streaming behaviour; the pre-fix code had no streamable_len",
         "find": "    end = max(settled - holdback, 0)\n", "replace": "    end = settled\n"},
        {"id": "streamable_len/streams-pending-bytes", "kind": "missing-logic", "origin": PRE_FIX,
         "summary": "a trailing incomplete UTF-8 sequence (U+FFFD) is streamed instead of held",
         "note": "reconstructed from the pre-fix streaming behaviour; the pre-fix code had no streamable_len",
         "find": "    settled = len(text.rstrip(\"\\ufffd\").encode(\"utf-8\"))\n",
         "replace": "    settled = len(data)\n"},
    ],
}


def streamable_len(text, holdback):
    """Return how many UTF-8 bytes at the start of text can be streamed now.

    A run of U+FFFD characters at the end of text is an incomplete UTF-8 sequence that
    the next token may complete, so it is never streamed. Of the bytes before that run,
    the last `holdback` bytes are also withheld, because they may start a stop sequence;
    the pending U+FFFD bytes do not count toward the holdback. The cut never splits a
    multi-byte character: it moves back to the start of that character. The result
    counts UTF-8 bytes and is never negative. Raises ValueError if holdback < 0.
    """
    if holdback < 0:
        raise ValueError("holdback must be non-negative")
    data = text.encode("utf-8")
    settled = len(text.rstrip("\ufffd").encode("utf-8"))
    end = max(settled - holdback, 0)
    while end < len(data) and (data[end] & 0xC0) == 0x80:
        end -= 1
    return end


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["", 0],
        ["", 3],
        ["ab\ufffd", 0],
        ["abc", 2],
        ["a\u00e9", 1],
        ["ab\ufffd", 3],
        ["abcd\ufffd", 3],
        ["abc", 0],
        ["abc", 3],
        ["abc", 4],
        ["\u20ac\u20ac", 1],
        ["\u20ac\u20ac", 2],
        ["x\ufffd\ufffd", 0],
        ["\ufffd", 1],
        ["abc", -1],
    ]
    alphabet = ["a", "b", " ", "\u00e9", "\u20ac", "\U0001f600", "z"]
    while len(cases) < n:
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 8)))
        if rng.random() < 0.3:
            text += "\ufffd" * rng.randint(1, 2)
        holdback = 0 if rng.random() < 0.4 else rng.randint(1, 5)
        cases.append([text, holdback])
    return cases[:n]
