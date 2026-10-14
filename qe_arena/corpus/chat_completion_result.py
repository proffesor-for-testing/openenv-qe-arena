"""Build the finish_reason and token usage of an OpenAI-style chat completion. Ported from Rust; adapted from ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm-cli/src/commands/serve.rs (MIT)."""

PRE_FIX = "ruvnet/RuVector@3e1d1886c50f2db817664648b95fab18e3e4c34b:crates/ruvllm-cli/src/commands/serve.rs"
META = {
    "entry": "chat_completion_result",
    "origin": "ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm-cli/src/commands/serve.rs",
    "tags": ["llm", "api", "accounting"],
    "domain": "software-engineering",
    "real_faults": [
        {"id": "chat_completion_result/always-stop", "kind": "constant-field", "origin": PRE_FIX,
         "summary": "finish_reason is always stop, even when max_tokens cut the reply or generation failed",
         "find": "    if cause == \"length\":\n        finish_reason = \"length\"\n"
                 "    elif cause in (\"stop\", \"end_of_sequence\", \"cancelled\"):\n        finish_reason = \"stop\"\n"
                 "    else:\n        raise ValueError(\"generation ended with an error\")\n",
         "replace": "    finish_reason = \"stop\"\n"},
        {"id": "chat_completion_result/whitespace-token-count", "kind": "placeholder-approximation",
         "origin": PRE_FIX, "summary": "usage counts whitespace-separated words, not tokens",
         "find": "    used = [prompt_tokens, completion_tokens]\n",
         "replace": "    used = [len(prompt.split()), len(completion.split())]\n"},
    ],
}


def chat_completion_result(cause, prompt, completion, prompt_tokens, completion_tokens):
    """Return {"finish_reason": ..., "usage": {...}} for a finished chat completion.

    cause says how generation ended. "length" (max_tokens reached) gives finish_reason
    "length"; "stop" (a stop sequence), "end_of_sequence" (an end-of-turn token) and
    "cancelled" give "stop"; "error" or any other cause raises ValueError. usage is
    {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c} where p and c
    are the tokenizer's counts prompt_tokens and completion_tokens; the prompt and
    completion texts do not affect usage. Raises ValueError if a count is negative.
    """
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if cause == "length":
        finish_reason = "length"
    elif cause in ("stop", "end_of_sequence", "cancelled"):
        finish_reason = "stop"
    else:
        raise ValueError("generation ended with an error")
    used = [prompt_tokens, completion_tokens]
    usage = {"prompt_tokens": used[0], "completion_tokens": used[1], "total_tokens": used[0] + used[1]}
    return {"finish_reason": finish_reason, "usage": usage}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["stop", "", "", 0, 0],
        ["length", "Hi", "Hello there", 1, 2],
        ["end_of_sequence", "Capital of France?", "Paris", 5, 1],
        ["cancelled", "a b", "c", 2, 1],
        ["error", "a", "b", 1, 1],
        ["timeout", "a", "b", 1, 1],
        ["stop", "one two three", "four five", 3, 2],
        ["length", "x", "y", -1, 0],
        ["stop", "x", "y", 0, -1],
        ["STOP", "x", "y", 1, 1],
        ["stop", "Hello, world!", "Bonjour", 4, 3],
        ["end_of_sequence", "a b c d", "", 9, 0],
    ]
    words = ["the", "cat", "sat", "Paris", "is", "x=1;", "ok", "no"]
    causes = ["stop", "stop", "end_of_sequence", "end_of_sequence", "length", "cancelled", "error"]
    while len(cases) < n:
        prompt = " ".join(rng.choice(words) for _ in range(rng.randint(0, 5)))
        completion = " ".join(rng.choice(words) for _ in range(rng.randint(0, 5)))
        p = len(prompt.split()) + (rng.randint(0, 3) if rng.random() < 0.6 else 0)
        c = len(completion.split()) + (rng.randint(0, 3) if rng.random() < 0.6 else 0)
        cases.append([rng.choice(causes), prompt, completion, p, c])
    return cases[:n]
