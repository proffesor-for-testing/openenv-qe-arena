#!/usr/bin/env python3
"""Build the CPython-WASI runtime the sandbox executes: python.wasm plus a reproducible stdlib zip.

    python scripts/build_wasi_runtime.py --out runtime            # downloads the pinned release
    python scripts/build_wasi_runtime.py --out runtime --asset python-3.14.7-wasi_sdk-24.zip

Steps:
  1. take brettcannon/cpython-wasi-build v3.14.7 (built with WASI SDK 24) and check its sha256;
  2. compile the stdlib with python.wasm itself, forcing unchecked-hash pycs (-f): timestamp pycs
     go stale after any copy and leak the host's mtimes into the guest, which changed startup fuel
     from 184 M to 1.18 G on another host;
  3. pack the pycs into one zip with sorted entries and fixed timestamps: a directory stdlib leaks
     the host's readdir order into the guest heap (a 25 k fuel difference across hosts).

Only the standard library and wasmtime-py are needed, so the Docker builder stage can run it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "v3.14.7"
ASSET = "python-3.14.7-wasi_sdk-24.zip"
ASSET_URL = f"https://github.com/brettcannon/cpython-wasi-build/releases/download/{RELEASE}/{ASSET}"
ASSET_SHA256 = "2e064d3fb8172471d39d741348efa722349c40b96301f69968dff714999c584b"
ZIP_NAME = "python314.zip"
# The output pins (python.wasm, the stdlib zip) live in qe_arena/runtime_pins.json, which the sandbox
# worker also checks before it starts. Same bytes on arm64 macOS and amd64 Linux.
DEFAULT_PINS = Path(__file__).resolve().parents[1] / "qe_arena" / "runtime_pins.json"
FIXED_DATE = (2020, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(asset: Path | None, workdir: Path) -> Path:
    if asset is None:
        asset = workdir / ASSET
        with urllib.request.urlopen(ASSET_URL, timeout=120) as response, asset.open("wb") as out:
            shutil.copyfileobj(response, out)
    actual = sha256(asset)
    if actual != ASSET_SHA256:
        raise SystemExit(f"{asset}: sha256 {actual} != pinned {ASSET_SHA256}")
    return asset


# Equivalent to `compileall -f -o 0 --invalidation-mode unchecked-hash`, but in sorted path order:
# compileall follows the host's readdir order, and marshal output depends on what was compiled before
# (one stdlib pyc differed between two build orders), so the order must not come from the filesystem.
COMPILE_ALL = """
import os, py_compile, sys
paths = []
for top, dirs, files in os.walk("/lib/python3.14"):
    paths += [os.path.join(top, f) for f in files if f.endswith(".py")]
failed = 0
for path in sorted(paths):
    try:
        py_compile.compile(path, doraise=True, optimize=0,
                           invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    except py_compile.PyCompileError:
        failed += 1  # test fixtures with deliberate syntax errors; compileall skips them too
print(len(paths), "sources,", failed, "skipped", file=sys.stderr)
"""


def compile_stdlib(root: Path) -> None:
    """Compile every stdlib module to an unchecked-hash pyc inside python.wasm itself."""
    import wasmtime as wt

    engine = wt.Engine(wt.Config())
    store = wt.Store(engine)
    wasi = wt.WasiConfig()
    wasi.argv = ["python", "-S", "-B", "-c", COMPILE_ALL]
    wasi.env = [("PYTHONHASHSEED", "0")]
    wasi.preopen_dir(str(root / "lib"), "/lib", wt.DirPerms.READ_WRITE, wt.FilePerms.READ_WRITE)
    wasi.inherit_stderr()
    store.set_wasi(wasi)
    linker = wt.Linker(engine)
    linker.define_wasi()
    module = wt.Module.from_file(engine, str(root / "python.wasm"))
    try:
        linker.instantiate(store, module).exports(store)["_start"](store)
    except wt.ExitTrap as exc:
        if exc.code != 0:
            raise SystemExit(f"compileall exited with {exc.code}") from exc


def pack(root: Path, out: Path) -> int:
    src = root / "lib" / "python3.14"
    entries = []
    for pyc in src.rglob("__pycache__/*.cpython-314.pyc"):
        rel = pyc.parent.parent.relative_to(src) / (pyc.name.split(".")[0] + ".pyc")
        entries.append((rel.as_posix(), pyc.read_bytes()))
    (out / "lib").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "lib" / ZIP_NAME, "w", zipfile.ZIP_STORED) as archive:
        for name, data in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=FIXED_DATE)
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset", type=Path, help="a local copy of the release zip (sha256 is still checked)")
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS, help="expected output digests")
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wasi-build-") as tmp:
        work = Path(tmp)
        asset = fetch(args.asset, work)
        root = work / "pyroot"
        with zipfile.ZipFile(asset) as archive:
            archive.extractall(root)
        compile_stdlib(root)
        modules = pack(root, args.out)
        shutil.copyfile(root / "python.wasm", args.out / "python.wasm")
    manifest = {
        "release": f"brettcannon/cpython-wasi-build {RELEASE}", "asset_sha256": ASSET_SHA256,
        "python.wasm": sha256(args.out / "python.wasm"),
        f"lib/{ZIP_NAME}": sha256(args.out / "lib" / ZIP_NAME), "modules": modules,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    if any(manifest[name] != digest for name, digest in pins.items()):
        print(f"runtime is not the pinned build: want {pins}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
