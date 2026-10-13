"""Normalize a chat transcript into strict user/assistant alternation. Adapted from proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/llm_providers/providers/anthropic_provider.py (MIT)."""

META = {
    "entry": "alternate_chat_roles",
    "origin": "proffesor-for-testing/sentinel-api-testing@704f47fbc4d2d75ed9261f9a602d76aea930d01b:sentinel_backend/llm_providers/providers/anthropic_provider.py",
    "tags": ["llm", "normalization", "lists"],
    "domain": "software-engineering",
}


def alternate_chat_roles(messages):
    """Return a new transcript whose roles strictly alternate and start and end with "user".

    messages is a list of {"role": "user" or "assistant", "content": str}; the input is
    not modified. Consecutive messages with the same role are merged into one message,
    joining the contents with "\n\n". After merging: if the first message is not from
    "user", {"role": "user", "content": "Continue."} is inserted at the front; then if
    the last message is not from "user", {"role": "user", "content": "Please respond."}
    is appended. An empty list returns [].
    """
    result = []
    for msg in messages:
        if result and result[-1]["role"] == msg["role"]:
            result[-1]["content"] += "\n\n" + msg["content"]
        else:
            result.append({"role": msg["role"], "content": msg["content"]})
    if result and result[0]["role"] != "user":
        result.insert(0, {"role": "user", "content": "Continue."})
    if result and result[-1]["role"] != "user":
        result.append({"role": "user", "content": "Please respond."})
    return result


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    def _m(role, content):
        return {"role": role, "content": content}

    u, a = "user", "assistant"
    cases = [
        [[]],
        [[_m(u, "hi")]],
        [[_m(a, "hi")]],
        [[_m(u, "q"), _m(a, "r")]],
        [[_m(u, "a"), _m(u, "b")]],
        [[_m(a, "a"), _m(a, "b"), _m(u, "c")]],
        [[_m(u, "a"), _m(a, "b"), _m(a, "c"), _m(u, "d")]],
        [[_m(a, "x"), _m(u, "y"), _m(u, "z"), _m(a, "w")]],
        [[_m(u, ""), _m(u, ""), _m(u, "")]],
    ]
    while len(cases) < n:
        cases.append([[_m(rng.choice([u, a]), rng.choice("abcdefg")) for _ in range(rng.randint(1, 6))]])
    return cases[:n]
