"""Resolve a GGUF model's end-of-sequence id and the ids that stop generation. Ported from Rust; adapted from ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs; ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/candle_backend.rs (MIT)."""

PRE_FIX = "ruvnet/RuVector@3e1d1886c50f2db817664648b95fab18e3e4c34b:crates/ruvllm/src/backends/candle_backend.rs"
META = {
    "entry": "gguf_special_ids",
    "origin": "ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs; ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/candle_backend.rs",
    "tags": ["llm", "tokenizer", "lookup"],
    "real_faults": [
        {"id": "gguf_special_ids/eos-from-vocab-guess", "kind": "wrong-sibling-constant", "origin": PRE_FIX,
         "summary": "the declared EOS id is ignored: a Qwen instruct model gets its end-of-text token, not end-of-turn",
         "note": "pre-fix behaviour (EOS guessed from the vocabulary) expressed through the fixed function's interface",
         "find": "    eos = declared(\"eos\")\n    if eos is None:\n        eos = guess\n",
         "replace": "    eos = guess\n"},
        {"id": "gguf_special_ids/stop-is-only-eos", "kind": "missing-logic", "origin": PRE_FIX,
         "summary": "generation stops only at the EOS id, never at end-of-turn tokens",
         "note": "pre-fix behaviour (stop on the single EOS id) expressed through the fixed function's interface",
         "find": "    stop = [vocab[t] for t in STOPS if t in vocab]\n    for name in (\"eos\", \"eot\", \"eom\"):\n"
                 "        value = declared(name)\n        if value is not None and value not in stop:\n"
                 "            stop.append(value)\n",
         "replace": "    stop = [] if eos is None else [eos]\n"},
    ],
}


def gguf_special_ids(metadata, vocab):
    """Return {"eos": id or None, "stop": [ids]} for a GGUF model.

    vocab maps token text to id. A declared id metadata["tokenizer.ggml.<x>_token_id"]
    (x is eos, eot or eom) counts only if it is an int (not a bool) with 0 <= id < len(vocab).
    eos is the declared eos id; without one, the id of the first of "</s>",
    "<end_of_text>", "<endoftext>", "<eot_id>" present in vocab; else None.
    stop lists the vocab ids of "<im_end>", "<endoftext>", "<eot_id>",
    "<end_of_text>", "</s>" (in that order, when present), then the declared eos,
    eot and eom ids, skipping any id already in the list.
    """
    def declared(x):
        value = metadata.get("tokenizer.ggml." + x + "_token_id")
        ok = isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(vocab)
        return value if ok else None

    candidates = ["</s>", "<end_of_text>", "<endoftext>", "<eot_id>"]
    guess = next((vocab[t] for t in candidates if t in vocab), None)
    STOPS = ["<im_end>", "<endoftext>", "<eot_id>", "<end_of_text>", "</s>"]
    eos = declared("eos")
    if eos is None:
        eos = guess
    stop = [vocab[t] for t in STOPS if t in vocab]
    for name in ("eos", "eot", "eom"):
        value = declared(name)
        if value is not None and value not in stop:
            stop.append(value)
    return {"eos": eos, "stop": stop}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    qwen = {"a": 0, "b": 1, "<endoftext>": 2, "<im_start>": 3, "<im_end>": 4}
    llama = {"<s>": 0, "</s>": 1, "a": 2, "b": 3}
    llama3 = {"<begin_of_text>": 0, "<end_of_text>": 1, "<eot_id>": 2, "a": 3, "b": 4, "c": 5}
    key = "tokenizer.ggml.{}_token_id".format
    cases = [
        [{}, {}],
        [{}, qwen],
        [{key("eos"): 4}, qwen],
        [{key("eos"): 2}, qwen],
        [{key("eos"): 5, key("eot"): 6, key("eom"): 99}, {"a": 0, "b": 1, "c": 2, "d": 3, "<my_eos>": 4,
                                                      "e": 5, "f": 6, "g": 7}],
        [{key("eos"): 1}, llama],
        [{key("eos"): 4}, llama],
        [{key("eos"): -1}, llama],
        [{key("eos"): True}, llama],
        [{key("eos"): "1"}, llama3],
        [{key("eot"): 2, key("eom"): 2}, llama3],
        [{key("eos"): 5}, llama3],
    ]
    vocabs = [qwen, llama, llama3, {"x": 0, "y": 1, "z": 2}]
    while len(cases) < n:
        vocab = rng.choice(vocabs)
        md = {}
        for x in ("eos", "eot", "eom"):
            if rng.random() < 0.5:
                md[key(x)] = rng.choice([rng.randint(-1, len(vocab) + 1), rng.randrange(len(vocab))])
        cases.append([md, vocab])
    return cases[:n]
