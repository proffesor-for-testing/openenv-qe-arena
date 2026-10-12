#!/usr/bin/env python3
"""Offline Ed25519 signing of task manifests (see qe_arena/manifest.py).

    # once, on a trusted machine: the private key never goes into the repo, the image or the server
    python scripts/sign_manifest.py keygen --private ~/.config/qe-arena/manifest_ed25519.pem \\
        --public qe_arena/keys/manifest_ed25519.pub [--passphrase-env QE_MANIFEST_PASSPHRASE]

    python scripts/sign_manifest.py sign --private KEY.pem --task qe_bugfix-d1:0 --task qe_mutant_kill-d3:7 \\
        --out manifests.jsonl
    python scripts/sign_manifest.py sign --private KEY.pem --task-ids all --seeds 0-63 --out manifests.jsonl

    python scripts/sign_manifest.py verify manifests.jsonl [--public qe_arena/keys/manifest_ed25519.pub]

verify recomputes every manifest from its (task_id, seed) and exits 1 if any signature or any
recomputed field disagrees.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from qe_arena import manifest  # noqa: E402
from qe_arena.tasks import TASK_IDS  # noqa: E402


def _passphrase(env: str | None) -> bytes | None:
    if not env:
        return None
    value = os.environ.get(env)
    if not value:
        raise SystemExit(f"{env} is not set")
    return value.encode()


def keygen(args: argparse.Namespace) -> int:
    for path in (args.private, args.public):
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite a key")
    key = Ed25519PrivateKey.generate()
    secret = _passphrase(args.passphrase_env)
    encryption = serialization.BestAvailableEncryption(secret) if secret else serialization.NoEncryption()
    args.private.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption))
    args.public.parent.mkdir(parents=True, exist_ok=True)
    args.public.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                          serialization.PublicFormat.SubjectPublicKeyInfo))
    print(f"key_id {manifest.key_id(key.public_key())}: private {args.private} (0600), public {args.public}")
    return 0


def _tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    pairs = []
    for item in args.task or []:
        task_id, _, seed = item.partition(":")
        pairs.append((task_id, int(seed)))
    if args.task_ids:
        ids = TASK_IDS if args.task_ids == "all" else args.task_ids.split(",")
        lo, _, hi = (args.seeds or "0").partition("-")
        seeds = range(int(lo), int(hi or lo) + 1)
        pairs += [(task_id, seed) for task_id in ids for seed in seeds]
    if not pairs:
        raise SystemExit("name tasks with --task ID:SEED or --task-ids ... --seeds A-B")
    return pairs


def sign(args: argparse.Namespace) -> int:
    key = serialization.load_pem_private_key(args.private.read_bytes(), password=_passphrase(args.passphrase_env))
    if not isinstance(key, Ed25519PrivateKey):
        raise SystemExit(f"{args.private} is not an Ed25519 private key")
    with args.out.open("w") as out:
        for task_id, seed in _tasks(args):
            out.write(json.dumps(manifest.sign(manifest.build_manifest(task_id, seed), key), sort_keys=True) + "\n")
    return 0


def verify(args: argparse.Namespace) -> int:
    key = manifest.load_public_key(args.public)
    failures = 0
    for line_no, line in enumerate(args.manifests.read_text().splitlines(), 1):
        envelope = json.loads(line)
        ok, reason = manifest.verify_task(envelope, key)
        m = envelope.get("manifest", {})
        print(f"{'OK  ' if ok else 'FAIL'} {m.get('task_id')}:{m.get('seed')} {reason}")
        failures += not ok
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--private", type=Path, required=True)
    k.add_argument("--public", type=Path, default=manifest.DEFAULT_PUBLIC_KEY)
    k.add_argument("--passphrase-env", help="encrypt the private key with the passphrase in this variable")
    s = sub.add_parser("sign")
    s.add_argument("--private", type=Path, required=True)
    s.add_argument("--passphrase-env")
    s.add_argument("--task", action="append", help="TASK_ID:SEED, repeatable")
    s.add_argument("--task-ids", help="comma-separated task IDs, or 'all'")
    s.add_argument("--seeds", help="A-B (inclusive) for --task-ids")
    s.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("manifests", type=Path)
    v.add_argument("--public", type=Path, default=manifest.DEFAULT_PUBLIC_KEY)
    args = parser.parse_args()
    return {"keygen": keygen, "sign": sign, "verify": verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
