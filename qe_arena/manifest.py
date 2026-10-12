"""Signed task manifests: a verifiable record of what graded a (task_id, seed).

Execution is deterministic (see ``sandbox``), so a task's hidden inputs, reference outputs and
their fuel are a pure function of (task_id, seed) plus the exact runtime, harness and corpus file.
A manifest pins all of those by sha256 (hashes only; no hidden value ever appears in one), signs
the canonical JSON with Ed25519, and lets anyone holding the public key recompute it from
(task_id, seed) and check both the signature and the content.

Offline only: the server never imports this module and never holds the private key. Keys and
signing live in ``scripts/sign_manifest.py``; the public key is ``qe_arena/keys/manifest_ed25519.pub``.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import sandbox, wasi_worker
from .tasks import Task, make_task

SCHEMA = "qe-arena-task-manifest/1"
DOMAIN = (SCHEMA + "\n").encode()  # signatures over manifests can never be replayed as anything else
PACKAGE = Path(__file__).resolve().parent
DEFAULT_PUBLIC_KEY = PACKAGE / "keys" / "manifest_ed25519.pub"


def canonical(obj: Any) -> bytes:
    """The one byte encoding that is hashed and signed: sorted keys, no whitespace, ASCII."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=True).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _grading_sources(task: Task) -> list[str]:
    p = task.private
    if task.family == "qe_bugfix":
        return [p["original"], p["buggy"]]
    return [p["original"]] + [m["source"] for m in p["mutants"]]


def manifest_for(task: Task) -> dict:
    p = task.private
    root = sandbox.runtime_root()
    reference = [[r.kind, r.value, r.fuel] for r in p["reference"]]
    return {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "seed": task.seed,
        "family": task.family,
        "difficulty": task.difficulty,
        "corpus": p["corpus"],
        "entry": p["entry"],
        "runtime": {
            "python.wasm": _file_sha(root / "python.wasm"),
            "lib/python314.zip": _file_sha(root / "lib" / "python314.zip"),
            "guest_harness.py": _file_sha(PACKAGE / "guest_harness.py"),
            "wasmtime": importlib.metadata.version("wasmtime"),
        },
        "corpus_file_sha256": _file_sha(PACKAGE / "corpus" / f"{p['corpus']}.py"),
        "hidden_inputs_count": len(p["inputs"]),
        "hidden_inputs_sha256": _sha(canonical(p["inputs"])),
        "reference_outputs_sha256": _sha(canonical(reference)),
        "grading_sources_sha256": _sha(canonical(_grading_sources(task))),
        "observation_sha256": _sha(canonical([task.prompt, task.files])),
        "budgets": {
            "fuel_per_call": sandbox.FUEL_PER_CALL,
            "module_fuel": sandbox.MODULE_FUEL,
            "startup_fuel": wasi_worker.STARTUP_FUEL,
            "harness_fuel": wasi_worker.HARNESS_FUEL,
            "mem_mb": sandbox.DEFAULT_MEM_MB,
        },
    }


def build_manifest(task_id: str, seed: int) -> dict:
    """Regenerate the task and describe it. Two calls with the same inputs return equal dicts."""
    return manifest_for(make_task(task_id, seed))


def key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return _sha(raw)[:16]


def sign(manifest: dict, private_key: Ed25519PrivateKey) -> dict:
    signature = private_key.sign(DOMAIN + canonical(manifest))
    return {"manifest": manifest, "key_id": key_id(private_key.public_key()),
            "signature": base64.b64encode(signature).decode("ascii")}


def verify_signature(envelope: dict, public_key: Ed25519PublicKey) -> bool:
    """True only for an envelope signed by this key over exactly this manifest."""
    try:
        if set(envelope) != {"manifest", "key_id", "signature"} or envelope["key_id"] != key_id(public_key):
            return False
        public_key.verify(base64.b64decode(envelope["signature"], validate=True),
                          DOMAIN + canonical(envelope["manifest"]))
        return True
    except (InvalidSignature, ValueError, TypeError, KeyError):
        return False


def verify_task(envelope: dict, public_key: Ed25519PublicKey | None = None) -> tuple[bool, str]:
    """Check the signature, then recompute the manifest from its (task_id, seed) and compare."""
    key = public_key if public_key is not None else load_public_key()
    if not verify_signature(envelope, key):
        return False, "bad signature"
    manifest = envelope["manifest"]
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        return False, "unknown schema"
    try:
        recomputed = build_manifest(manifest["task_id"], manifest["seed"])
    except (KeyError, ValueError, RuntimeError) as exc:
        return False, f"cannot rebuild the task: {exc}"
    if recomputed != manifest:
        changed = sorted(k for k in set(recomputed) | set(manifest) if recomputed.get(k) != manifest.get(k))
        return False, "recomputed manifest differs in: " + ", ".join(changed)
    return True, "ok"


def load_public_key(path: Path = DEFAULT_PUBLIC_KEY) -> Ed25519PublicKey:
    if not path.exists():
        raise FileNotFoundError(f"{path}: no manifest public key; create one with scripts/sign_manifest.py keygen")
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{path} is not an Ed25519 public key")
    return key
