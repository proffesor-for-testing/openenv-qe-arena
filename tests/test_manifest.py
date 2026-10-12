"""Signed task manifests: round trip, tampering, wrong keys, recomputation, and the CLI."""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from qe_arena import manifest
from qe_arena.tasks import make_task

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("qe_bugfix-d1", 0), ("qe_bugfix-d3", 1), ("qe_mutant_kill-d1", 0), ("qe_mutant_kill-d3", 2)]
HEX64 = re.compile(r"[0-9a-f]{64}")


@pytest.fixture(scope="module")
def built():
    return {(task_id, seed): manifest.build_manifest(task_id, seed) for task_id, seed in TASKS}


@pytest.fixture(scope="module")
def key():
    return Ed25519PrivateKey.generate()


def _leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, path + (k,))
    else:
        yield path, obj


def test_recompute_equals_the_manifest_for_several_tasks(built):
    for (task_id, seed), m in built.items():
        assert manifest.build_manifest(task_id, seed) == m
        assert (m["task_id"], m["seed"], m["schema"]) == (task_id, seed, manifest.SCHEMA)


def test_manifest_holds_hashes_and_budgets_never_hidden_values(built):
    for (task_id, seed), m in built.items():
        task = make_task(task_id, seed)
        text = manifest.canonical(m).decode()
        for value in task.private["inputs"][:10] + [r.value for r in task.private["reference"][:10]]:
            if isinstance(value, (list, dict)) and json.dumps(value) not in ("[]", "{}"):
                assert json.dumps(value, separators=(",", ":")) not in text
        for key in ("hidden_inputs_sha256", "reference_outputs_sha256", "grading_sources_sha256",
                    "observation_sha256", "corpus_file_sha256"):
            assert HEX64.fullmatch(m[key]), key
        assert all(HEX64.fullmatch(v) for k, v in m["runtime"].items() if k != "wasmtime")
        assert m["runtime"]["wasmtime"] == "47.0.1"
        assert m["budgets"] == {"fuel_per_call": 100_000_000, "module_fuel": 200_000_000,
                                "startup_fuel": 3_000_000_000, "harness_fuel": 1_000_000_000, "mem_mb": 256}
        assert m["hidden_inputs_count"] == len(task.private["inputs"])


def test_signature_round_trip_and_full_verification(built, key):
    m = built[TASKS[0]]
    envelope = manifest.sign(m, key)
    assert manifest.verify_signature(envelope, key.public_key())
    assert manifest.verify_task(envelope, key.public_key()) == (True, "ok")
    assert manifest.verify_signature(json.loads(json.dumps(envelope)), key.public_key())  # survives JSON


def test_every_tampered_field_fails_the_signature(built, key):
    envelope = manifest.sign(built[TASKS[2]], key)
    for path, value in _leaves(envelope["manifest"]):
        forged = copy.deepcopy(envelope)
        node = forged["manifest"]
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value + 1 if isinstance(value, int) else str(value) + "x"
        assert not manifest.verify_signature(forged, key.public_key()), path
    extra = copy.deepcopy(envelope)
    extra["manifest"]["note"] = "added"
    assert not manifest.verify_signature(extra, key.public_key())


def test_wrong_key_or_malformed_envelope_fails(built, key):
    envelope = manifest.sign(built[TASKS[1]], key)
    other = Ed25519PrivateKey.generate()
    assert not manifest.verify_signature(envelope, other.public_key())
    relabelled = dict(envelope, key_id=manifest.key_id(other.public_key()))
    assert not manifest.verify_signature(relabelled, other.public_key())
    assert manifest.verify_task(envelope, other.public_key()) == (False, "bad signature")
    for broken in (dict(envelope, signature="!!notbase64!!"), dict(envelope, signature=""),
                   {k: v for k, v in envelope.items() if k != "key_id"}, dict(envelope, extra=1)):
        assert not manifest.verify_signature(broken, key.public_key())


def test_a_validly_signed_but_wrong_manifest_fails_recomputation(built, key):
    # The signer's key is not proof of content: a manifest signed over a wrong reference hash (or a
    # different runtime) must still fail, because verify_task rebuilds the task and compares.
    for field, value in (("reference_outputs_sha256", "0" * 64), ("seed", 1), ("budgets", {"fuel_per_call": 1})):
        lie = dict(built[TASKS[0]], **{field: value})
        ok, reason = manifest.verify_task(manifest.sign(lie, key), key.public_key())
        assert not ok and reason != "bad signature", field
    runtime = dict(built[TASKS[0]], runtime=dict(built[TASKS[0]]["runtime"], wasmtime="48.0.0"))
    ok, reason = manifest.verify_task(manifest.sign(runtime, key), key.public_key())
    assert not ok and "runtime" in reason


def test_missing_public_key_says_how_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="keygen"):
        manifest.load_public_key(tmp_path / "absent.pub")


def test_cli_keygen_sign_verify_and_tamper(tmp_path):
    cli = [sys.executable, str(ROOT / "scripts" / "sign_manifest.py")]
    private, public, out = tmp_path / "k.pem", tmp_path / "k.pub", tmp_path / "m.jsonl"
    run = lambda *a: subprocess.run(cli + list(a), capture_output=True, text=True, cwd=ROOT)  # noqa: E731
    made = run("keygen", "--private", str(private), "--public", str(public))
    assert made.returncode == 0, made.stderr
    assert stat.S_IMODE(os.stat(private).st_mode) == 0o600
    assert run("keygen", "--private", str(private), "--public", str(public)).returncode != 0  # no overwrite
    signed = run("sign", "--private", str(private), "--task", "qe_bugfix-d1:0", "--task", "qe_mutant_kill-d1:0",
                 "--out", str(out))
    assert signed.returncode == 0, signed.stderr
    good = run("verify", str(out), "--public", str(public))
    assert good.returncode == 0 and good.stdout.count("OK") == 2, good.stdout
    lines = out.read_text().splitlines()
    envelope = json.loads(lines[1])
    envelope["manifest"]["hidden_inputs_count"] += 1
    out.write_text(lines[0] + "\n" + json.dumps(envelope) + "\n")
    bad = run("verify", str(out), "--public", str(public))
    assert bad.returncode == 1 and "FAIL qe_mutant_kill-d1:0 bad signature" in bad.stdout


def test_the_server_never_loads_the_manifest_module():
    probe = ("import sys, qe_arena.app, qe_arena.environment, qe_arena.tasks; "
             "print('qe_arena.manifest' in sys.modules)")
    env = dict(os.environ, QE_ARENA_WASI_WORKERS="1")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT, env=env)
    assert result.stdout.strip() == "False", result.stderr[-2000:]
