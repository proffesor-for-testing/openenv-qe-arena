"""Fetch and pin the public data behind the qe_skill desk families (level 4).

Run by a maintainer, with network: ``python scripts/fetch_desk_data.py`` pins sources that are not pinned yet.
It writes gzipped snapshots to ``qe_arena/skills/data/`` and records each source URL, licence, retrieval date and
SHA-256 in ``qe_arena/skills/data/PROVENANCE.json``. Task generation reads only those pinned files, never the network, so a
seed rebuilds the same task on any host. ``--check`` re-hashes the pinned files against PROVENANCE.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "qe_arena" / "skills" / "data"
US_GOV = "Public domain: a work of the United States Government (17 U.S.C. 105)."
NWIS_SITES = ("05331000", "15515500", "01646500", "12340500", "06191500",
              "14105700", "08313000", "02087500", "04264331", "03193000")
NWIS_URL = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={site}&parameterCd=00060&statCd=00003"
            "&startDT=2014-10-01&endDT=2026-10-26")
KEV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
TREASURY_YEARS = range(2018, 2027)
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
                "{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")
# USGS ComCat slices with recent automatic solutions, mixed magnitude types and non-earthquake events.
QUAKE_SLICES = {
    "northern_california_2026a": ("2026-04-01", "2026-07-15", 36.5, 39.5, -123.5, -120.5),
    "northern_california_2026b": ("2026-07-15", "2026-10-27", 36.5, 39.5, -123.5, -120.5),
    "nevada_2026a": ("2026-04-01", "2026-07-15", 36.0, 41.0, -120.0, -114.0),
    "nevada_2026b": ("2026-07-15", "2026-10-27", 36.0, 41.0, -120.0, -114.0),
}
QUAKE_URL = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time&starttime={0}&endtime={1}"
             "&minlatitude={2}&maxlatitude={3}&minlongitude={4}&maxlongitude={5}&minmagnitude=1.5")
GML_SERIES = (("co2", "brw"), ("co2", "mlo"), ("co2", "smo"), ("co2", "spo"), ("ch4", "brw"), ("ch4", "mlo"))
GML_URL = ("https://gml.noaa.gov/aftp/data/trace_gases/{gas}/in-situ/surface/txt/"
           "{gas}_{site}_surface-insitu_1_ccgg_MonthlyData.txt")
GML_TERMS = (US_GOV + " NOAA Global Monitoring Laboratory data are made freely available to the public; GML asks "
             "users to give fair credit and to cite the data providers (file header, 'USE OF NOAA GML DATA').")


def sources() -> list[dict[str, str]]:
    out = [{"path": f"nwis/{site}.rdb.gz", "url": NWIS_URL.format(site=site), "publisher": "U.S. Geological Survey",
            "dataset": f"NWIS daily values, discharge (00060), daily mean, site {site}", "licence": US_GOV}
           for site in NWIS_SITES]
    out.append({"path": "kev/known_exploited_vulnerabilities.csv.gz", "url": KEV_URL,
                "publisher": "Cybersecurity and Infrastructure Security Agency",
                "dataset": "Known Exploited Vulnerabilities Catalog", "licence": US_GOV})
    out += [{"path": f"treasury/par_yield_curve_{year}.csv.gz", "url": TREASURY_URL.format(year=year),
             "publisher": "U.S. Department of the Treasury",
             "dataset": f"Daily Treasury Par Yield Curve Rates, {year}", "licence": US_GOV} for year in TREASURY_YEARS]
    out += [{"path": f"quakes/{name}.csv.gz", "url": QUAKE_URL.format(*box), "publisher": "U.S. Geological Survey",
             "dataset": f"ANSS ComCat earthquake catalogue (FDSN event CSV), M1.5+, {name}", "licence": US_GOV}
            for name, box in QUAKE_SLICES.items()]
    out += [{"path": f"gml/{gas}_{site}_monthly.txt.gz", "url": GML_URL.format(gas=gas, site=site),
             "publisher": "NOAA Global Monitoring Laboratory",
             "dataset": f"In-situ {gas.upper()} monthly means, {site.upper()} observatory", "licence": GML_TERMS}
            for gas, site in GML_SERIES]
    return out


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "openenv-qe-arena data pin"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https sources above
        return response.read()


def pin(refresh: bool) -> None:
    """Fetch sources that are not pinned yet (all of them with ``refresh``); a pinned snapshot is never re-fetched
    silently, because some sources (USGS) stamp the retrieval time into the file."""
    known = {r["path"]: r for r in json.loads((DATA / "PROVENANCE.json").read_text())} \
        if (DATA / "PROVENANCE.json").exists() else {}
    records = []
    for source in sources():
        if not refresh and source["path"] in known and (DATA / source["path"]).exists():
            records.append(known[source["path"]])
            continue
        raw = fetch(source["url"])
        target = DATA / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(raw, mtime=0))
        records.append({**source, "retrieved": dt.date.today().isoformat(), "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        print(f"pinned {source['path']}: {len(raw)} bytes")
    (DATA / "PROVENANCE.json").write_text(json.dumps(records, indent=2) + "\n")


def check() -> int:
    bad = 0
    for record in json.loads((DATA / "PROVENANCE.json").read_text()):
        raw = gzip.decompress((DATA / record["path"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            print(f"MISMATCH {record['path']}")
            bad += 1
    return bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify pinned files against PROVENANCE.json")
    parser.add_argument("--refresh", action="store_true", help="re-fetch sources that are already pinned")
    args = parser.parse_args()
    if args.check:
        sys.exit(1 if check() else 0)
    pin(args.refresh)
