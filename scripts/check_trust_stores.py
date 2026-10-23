"""Reject the distrusted emSign G3 root in every Python image trust store."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import ssl
import sys

import certifi
import pip._vendor.certifi as pip_certifi

BAD_ROOT_SHA256 = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"


def main() -> int:
    os_bundle = ssl.get_default_verify_paths().cafile
    if not os_bundle:
        raise RuntimeError("No independently readable OS CA bundle")
    stores = {"application": certifi.where(), "pip_vendor": pip_certifi.where(), "os": os_bundle}
    result = {}
    for name, path in stores.items():
        roots = ssl.create_default_context(cafile=path).get_ca_certs(binary_form=True)
        result[name] = {
            "root_count": len(roots),
            "distrusted_g3_present": any(hashlib.sha256(root).hexdigest() == BAD_ROOT_SHA256 for root in roots),
        }
    versions = {name: version(name) for name in ("certifi", "pip")}
    version_ok = (
        tuple(map(int, versions["certifi"].split("."))) >= (2026, 10, 19)
        and tuple(map(int, versions["pip"].split("."))) >= (26, 2, 2)
    )
    print(json.dumps({"python": sys.version.split()[0], "versions": versions, "stores": result}, sort_keys=True))
    return int(not version_ok or any(not store["root_count"] or store["distrusted_g3_present"] for store in result.values()))


if __name__ == "__main__":
    raise SystemExit(main())
