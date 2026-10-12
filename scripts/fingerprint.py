#!/usr/bin/env python3
"""Cross-host determinism fingerprint of the sandbox.

SHA-256 over (outcome, value, fuel) for every corpus function run on 64 seeded inputs, plus the
total fuel. The same digest on two hosts means bit-identical execution and metering there.
tests/test_sandbox.py pins it, so a change to the runtime, the harness or the corpus shows up.

    python scripts/fingerprint.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qe_arena import corpus, sandbox  # noqa: E402

INPUT_SEED = 0
INPUTS = 64


def corpus_fingerprint() -> tuple[str, int, int]:
    """Return (first 16 hex of the digest, total fuel, number of calls)."""
    digest, total, calls = hashlib.sha256(), 0, 0
    for name in sorted(corpus.ENTRIES):
        item = corpus.ENTRIES[name]
        inputs = corpus.gen_inputs(name, INPUT_SEED, INPUTS)
        results = sandbox.run_calls(item.function_source, item.entry, inputs)
        digest.update(json.dumps([name, [(r.kind, r.value, r.fuel) for r in results]], sort_keys=True).encode())
        total += sum(r.fuel for r in results)
        calls += len(results)
    return digest.hexdigest()[:16], total, calls


def runtime_digests() -> dict[str, str]:
    root = sandbox.runtime_root()
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("python.wasm", "lib/python314.zip")}


def main() -> int:
    fingerprint, total, calls = corpus_fingerprint()
    print(json.dumps({"fingerprint": fingerprint, "total_fuel": total, "calls": calls,
                      "runtime": runtime_digests()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
