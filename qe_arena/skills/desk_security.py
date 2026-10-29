"""cybersecurity desk family (level 4): a vendor exposure report from a real CISA KEV catalogue snapshot.

See ``desk_science`` for what a desk family is. The input is the whole Known Exploited Vulnerabilities catalogue
(about 1 MB of CSV with quoted multi-line-safe fields), so the model has to parse it with a script.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import median, median_low

from . import Built, Family
from ._common import close, rows, write_csv

DOMAIN = "cybersecurity"
LEVEL = 4
SNAPSHOT = Path(__file__).resolve().parent / "data" / "kev" / "known_exploited_vulnerabilities.csv.gz"
FILE = "known_exploited_vulnerabilities.csv"
HEADER = ["vendor", "added", "ransomware_known", "overdue", "median_days_to_due", "top_cwe"]
FIRST, LAST = dt.date(2022, 1, 1), dt.date(2026, 9, 30)  # the window end is chosen in [FIRST + 1 year, LAST]


@lru_cache(maxsize=1)
def _snapshot() -> str:
    return gzip.decompress(SNAPSHOT.read_bytes()).decode("utf-8")


@lru_cache(maxsize=1)
def _dates() -> tuple[tuple[dt.date, dt.date], ...]:
    """(dateAdded, dueDate) of every catalogue entry, for choosing request dates that make the rules bind."""
    return tuple((dt.date.fromisoformat(e["dateAdded"]), dt.date.fromisoformat(e["dueDate"]))
                 for e in csv.DictReader(io.StringIO(_snapshot())))


@lru_cache(maxsize=4)
def _entries(text: str) -> tuple[dict[str, str], ...]:
    """Parsed catalogue rows; cached because every trap check of a build re-reads the same 1 MB text."""
    return tuple(csv.DictReader(io.StringIO(text)))


def _cwes(field: str) -> list[str]:
    return [part.strip() for part in field.split(",") if part.strip().startswith("CWE-")]


def _cwe_number(cwe: str) -> int:
    tail = cwe.split("-", 1)[1]
    return int(tail) if tail.isdigit() else 10**9


def _kev_build(level: int, rng: random.Random) -> Built:
    # The window ends on a day the catalogue added entries and as_of is one of the in-scope due dates, so the
    # inclusive-end and strictly-before rules change the answer instead of being decorative.
    end = rng.choice(sorted({a for a, _ in _dates() if FIRST + dt.timedelta(days=365) <= a <= LAST}))
    start = end - dt.timedelta(days=rng.randint(90, 365))
    as_of = rng.choice(sorted({due for added, due in _dates() if start <= added <= end}))
    request = {"added_from": start.isoformat(), "added_to": end.isoformat(), "as_of": as_of.isoformat(),
               "top_vendors": rng.randint(5, 10)}
    files = {FILE: _snapshot(), "request.json": json.dumps(request, indent=2) + "\n"}
    statement = "\n".join([
        f"The security team wants a vendor exposure summary from {FILE}, a snapshot of CISA's Known Exploited "
        "Vulnerabilities catalogue (one row per CVE; fields are CSV-quoted and some contain commas). request.json "
        "gives the reporting window, the as-of date and how many vendors to list.",
        "",
        "Rules:",
        "- An entry is in scope when its dateAdded falls in the window, both ends included (added_from and "
        "added_to are dates, YYYY-MM-DD).",
        "- Group in-scope entries by vendorProject, with leading and trailing spaces removed. Rank vendors by "
        "their number of in-scope entries, highest first, breaking ties by vendor name in ascending order, and keep "
        "the first top_vendors.",
        "- ransomware_known counts the vendor's in-scope entries whose knownRansomwareCampaignUse is Known.",
        "- overdue counts the vendor's in-scope entries whose dueDate is strictly before as_of.",
        "- median_days_to_due is the median of (dueDate - dateAdded) in days over the vendor's in-scope entries; "
        "with an even number of entries it is the mean of the two middle values. Give it with one decimal place.",
        "- top_cwe is the CWE that occurs in the most of the vendor's in-scope entries. The cwes field may list "
        "several CWEs separated by commas, and each listed CWE counts once for that entry; entries with an empty "
        "field add nothing. Break ties by the lowest CWE number (CWE-20 comes before CWE-119). Write NONE when no "
        "entry lists a CWE.",
        "",
        "Write vendor_report.csv with exactly this header line: " + ",".join(HEADER) + ", then one row per listed "
        "vendor in rank order. The catalogue is too long to print, so process it with a script.",
    ])
    built = Built(files=files, statement=statement, output="vendor_report.csv", reference=None)
    built.reference = rows(_kev_solve(files, level, None))
    return built


def _kev_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    start, end = dt.date.fromisoformat(request["added_from"]), dt.date.fromisoformat(request["added_to"])
    as_of = dt.date.fromisoformat(request["as_of"])
    by_vendor: dict[str, list[dict[str, str]]] = {}
    for entry in _entries(files[FILE]):
        added = dt.date.fromisoformat(entry["dateAdded"])
        if start <= added <= end and not (variant == "end_exclusive" and added == end):
            by_vendor.setdefault(entry["vendorProject"].strip(), []).append(entry)
    ranked = sorted(by_vendor.items(), key=lambda kv: (-len(kv[1]), kv[0]))[: request["top_vendors"]]
    records = []
    for vendor, entries in ranked:
        ransom = sum(1 for e in entries if e["knownRansomwareCampaignUse"] == "Known" or variant == "ransomware_all")
        due = [dt.date.fromisoformat(e["dueDate"]) for e in entries]
        overdue = sum(1 for d in due if d < as_of or (variant == "overdue_inclusive" and d == as_of))
        gaps = [(d - dt.date.fromisoformat(e["dateAdded"])).days for d, e in zip(due, entries)]
        mid = median_low(gaps) if variant == "median_lower" else median(gaps)
        counts: Counter[str] = Counter()
        for e in entries:
            listed = _cwes(e["cwes"])
            counts.update(set(listed[:1] if variant == "cwe_first_only" else listed))
        if variant == "cwe_lexicographic":
            top = min(counts, key=lambda c: (-counts[c], c)) if counts else "NONE"
        else:
            top = min(counts, key=lambda c: (-counts[c], _cwe_number(c), c)) if counts else "NONE"
        records.append([vendor, len(entries), ransom, overdue, f"{mid:.1f}", top])
    return write_csv(HEADER, records)


def _kev_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    first = text.splitlines()[0].strip() if text.strip() else ""
    out = rows(text)
    if first != ",".join(HEADER) or len(out) != len(reference):
        return [False]
    results = []
    for got, ref in zip(out, reference):
        got = {k: (v or "").strip() for k, v in got.items() if k is not None}
        try:
            mid = close(float(got["median_days_to_due"]), float(ref["median_days_to_due"]), 0.05)
        except ValueError:
            mid = False
        results += [got["vendor"] == ref["vendor"], got["added"] == ref["added"],
                    got["ransomware_known"] == ref["ransomware_known"], got["overdue"] == ref["overdue"], mid,
                    got["top_cwe"] == ref["top_cwe"]]
    return results


FAMILIES = (
    Family("kev", DOMAIN, _kev_build, _kev_check, _kev_solve,
           # median_lower and cwe_first_only are solver variants too, but the real catalogue changes the answer
           # under them on only 18% and 25% of requests, so they are not required to fail on every seed.
           {trap: (LEVEL,) for trap in ("end_exclusive", "ransomware_all", "overdue_inclusive", "cwe_lexicographic")},
           levels=(LEVEL,)),
)
