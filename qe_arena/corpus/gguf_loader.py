"""Choose the weight loader for a GGUF model file from its metadata. Ported from Rust; adapted from ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs (MIT)."""

PRE_FIX = "ruvnet/RuVector@3e1d1886c50f2db817664648b95fab18e3e4c34b:crates/ruvllm/src/backends/candle_backend.rs"
META = {
    "entry": "gguf_loader",
    "origin": "ruvnet/RuVector@32033a8bc0d4d92e0ea644fce98eef4bb5c6bfef:crates/ruvllm/src/backends/gguf_arch/mod.rs",
    "tags": ["llm", "dispatch", "validation"],
    "real_faults": [
        {"id": "gguf_loader/no-split-check", "kind": "missing-guard", "origin": PRE_FIX,
         "summary": "a split (multi-part) GGUF is not rejected",
         "find": "    if isinstance(parts, int) and parts > 1:\n        raise ValueError(\"split GGUF files are not supported\")\n",
         "replace": ""},
        {"id": "gguf_loader/qwen-rejected", "kind": "allowlist-too-narrow", "origin": PRE_FIX,
         "summary": "only llama-family layouts load; qwen2 and qwen3 are rejected",
         "find": "    if name == \"qwen2\":\n        return \"qwen2\"\n    if name == \"qwen3\":\n        return \"qwen3\"\n",
         "replace": ""},
        {"id": "gguf_loader/substring-match", "kind": "allowlist-too-wide", "origin": PRE_FIX,
         "summary": "any architecture containing llama or mistral loads as llama",
         "find": "    if name in (\"\", \"llama\", \"mistral\"):\n",
         "replace": "    if name in (\"\", \"llama\", \"mistral\") or \"llama\" in name or \"mistral\" in name:\n"},
    ],
}


def gguf_loader(metadata):
    """Return the weight loader ("llama", "qwen2" or "qwen3") for a GGUF file's metadata.

    If metadata["split.count"] is an int greater than 1, the model is split into several
    files, which is not supported: raise ValueError. The architecture is
    metadata["general.architecture"]; a missing key or a non-string value counts as "".
    It is matched case-insensitively and exactly: "", "llama" and "mistral" use the
    "llama" loader, "qwen2" uses "qwen2" and "qwen3" uses "qwen3". Every other
    architecture (for example "qwen35", "qwen3moe", "llama4", "codellama") raises
    ValueError naming it.
    """
    parts = metadata.get("split.count")
    if isinstance(parts, int) and parts > 1:
        raise ValueError("split GGUF files are not supported")
    arch = metadata.get("general.architecture", "")
    name = arch.lower() if isinstance(arch, str) else ""
    if name in ("", "llama", "mistral"):
        return "llama"
    if name == "qwen2":
        return "qwen2"
    if name == "qwen3":
        return "qwen3"
    raise ValueError("GGUF architecture " + repr(arch) + " is not supported")


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    arches = ["", "llama", "LLaMA", "mistral", "Mistral", "qwen2", "Qwen2", "qwen3", "QWEN3", "qwen35",
              "qwen3moe", "qwen3next", "llama4", "codellama", "mistral-nemo", "phi3", "gemma2"]
    cases = [
        [{}],
        [{"general.architecture": "LLaMA"}],
        [{"general.architecture": "qwen2"}],
        [{"general.architecture": "qwen3"}],
        [{"general.architecture": "qwen35"}],
        [{"general.architecture": "llama4"}],
        [{"general.architecture": "qwen2", "split.count": 3}],
        [{"split.count": 1}],
        [{"split.count": 2}],
        [{"general.architecture": 7}],
        [{"general.architecture": "phi3", "split.count": 0}],
        [{"general.architecture": "mistral", "split.count": 2}],
        [{"general.architecture": "Qwen2"}],
        [{"general.architecture": "codellama"}],
        [{"general.architecture": "Mistral-Nemo"}],
    ]
    while len(cases) < n:
        md = {}
        if rng.random() < 0.9:
            md["general.architecture"] = rng.choice(arches)
        if rng.random() < 0.35:
            md["split.count"] = rng.choice([0, 1, 1, 2, 3, 5])
        cases.append([md])
    return cases[:n]
