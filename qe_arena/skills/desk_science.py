"""natural-science desk family (level 4): water-year statistics from a real USGS daily-discharge export.

Desk families copy the shape of the arena's private tasks rather than their topics: a long prose brief, a real
public data file too large to read with ``cat`` (a command's output is cut at 8 KiB), several method rules that a
quick reading gets wrong, and one CSV with a fixed header. The data are pinned snapshots in ``data/`` (sources and
licences in ``data/PROVENANCE.json``); generation never touches the network.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import random
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from . import Built, Family
from ._common import close, rows, write_csv

DOMAIN = "natural-science"
LEVEL = 4
DATA = Path(__file__).resolve().parent / "data" / "nwis"
CFS = {"m3/s": 0.028316846592, "ML/day": 0.028316846592 * 86400 / 1000}
HEADER = ["water_year", "approved_days", "mean_flow", "low_flow", "peak_date"]
TOLERANCE = 0.0006  # three decimal places
START_YEARS = (2020, 2021, 2022, 2023)
MIN_DAYS = (300, 330, 355)
LOW_WINDOWS = (3, 7, 30)


@lru_cache(maxsize=None)
def _export(site: str) -> tuple[list[str], list[str]]:
    """(comment and header lines, data lines) of the pinned export for ``site``."""
    lines = gzip.decompress((DATA / f"{site}.rdb.gz").read_bytes()).decode("utf-8").splitlines()
    head = [line for line in lines if not line.startswith("USGS\t")]
    return head, [line for line in lines if line.startswith("USGS\t")]


def _sites() -> list[str]:
    return sorted(path.name.split(".")[0] for path in DATA.glob("*.rdb.gz"))


@lru_cache(maxsize=8)
def _parse(text: str) -> tuple[tuple[dt.date, float | None, str], ...]:
    """(date, cfs or None, qualifier) per data row of an RDB export; comment and type rows are skipped."""
    out = []
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    for line in lines[2:]:  # column names, then the RDB column-type row
        cells = line.split("\t")
        try:
            value = float(cells[3])
        except ValueError:  # blank, or a remark such as Ice or Eqp in place of a value
            value = None
        out.append((dt.date.fromisoformat(cells[2]), value, cells[4].strip()))
    return tuple(out)


def _water_year(day: dt.date, by_start: bool = False) -> int:
    year = day.year + (1 if day.month >= 10 else 0)
    return year - 1 if by_start else year


def _low_flow(good: dict[dt.date, float], n: int) -> float | None:
    """Lowest mean over n consecutive calendar days that are all in ``good``, or None."""
    best, run, previous = None, [], None
    for day in sorted(good):
        if previous is None or day - previous != dt.timedelta(days=1):
            run = []
        run.append(good[day])
        previous = day
        if len(run) >= n:
            mean = fmean(run[-n:])
            best = mean if best is None or mean < best else best
    return best


def _streamflow_build(level: int, rng: random.Random) -> Built:
    site = rng.choice(_sites())
    start = dt.date(rng.choice(START_YEARS), 1, 1)
    request = {"site": site, "unit": rng.choice(sorted(CFS)), "min_approved_days": rng.choice(MIN_DAYS),
               "low_flow_window_days": rng.choice(LOW_WINDOWS)}
    head, data = _export(site)
    body = [line for line in data if dt.date.fromisoformat(line.split("\t")[2]) >= start]
    name = f"usgs_{site}_daily.rdb"
    files = {name: "\n".join([*head, *body]) + "\n", "request.json": json.dumps(request, indent=2) + "\n"}
    n, unit, need = request["low_flow_window_days"], request["unit"], request["min_approved_days"]
    statement = "\n".join([
        f"A hydrology group needs annual flow statistics for the USGS gauge in {name}, a tab-separated (RDB) export "
        "of daily mean discharge in cubic feet per second. Lines starting with # are comments; the first other "
        "line names the columns, and the next one gives RDB column types, not data. The last column of each data "
        "row is the value's qualification code, explained in the file's comments. request.json gives the settings "
        "to use.",
        "",
        "Rules:",
        "- Work by USGS water year: water year Y runs from 1 October of Y-1 to 30 September of Y and is labelled by "
        "the year in which it ends.",
        "- A day counts only if it is approved: its code begins with A (estimated approved values such as A:e "
        "count; provisional P values do not). Days missing from the file, and days whose value is not a number "
        "(such as Ice), do not count either.",
        f"- Convert every value from ft3/s to {unit} (1 ft3/s = 0.028316846592 m3/s; 1 ML = 1000 m3; a day is "
        "86400 s).",
        f"- mean_flow is the mean of the approved daily values in the water year. low_flow is the lowest mean over "
        f"{n} consecutive calendar days that all lie in that water year and are all approved. peak_date is the "
        "date (YYYY-MM-DD) of the highest approved daily value; on a tie take the earliest date.",
        f"- A water year with fewer than {need} approved days is incomplete: give its approved_days but leave "
        "mean_flow, low_flow and peak_date empty. Leave low_flow empty too when no qualifying window exists.",
        "",
        "Write water_years.csv with exactly this header line: " + ",".join(HEADER) + ". Add one row for every "
        "water year that has at least one row in the file, in any order. Round mean_flow and low_flow to 3 decimal "
        f"places, in {unit}. The file is too long to print in full, so process it with a script.",
    ])
    built = Built(files=files, statement=statement, output="water_years.csv", reference=None)
    built.reference = rows(_streamflow_solve(files, level, None))
    return built


def _streamflow_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    export = next(text for name, text in files.items() if name.endswith(".rdb"))
    factor = 1.0 if variant == "no_conversion" else CFS[request["unit"]]
    n, need = request["low_flow_window_days"], request["min_approved_days"]
    years: dict[int, dict[dt.date, float | None]] = {}
    for day, value, code in _parse(export):
        label = day.year if variant == "calendar_year" else _water_year(day, variant == "label_by_start")
        approved = code.startswith("A") or (variant == "keep_provisional" and code.startswith("P"))
        if variant == "drop_estimated" and code.endswith(":e"):
            approved = False
        years.setdefault(label, {})[day] = value * factor if approved and value is not None else None
    records = []
    for label, days in sorted(years.items()):
        good = {day: v for day, v in days.items() if v is not None}
        if len(good) < need and variant != "ignore_completeness":
            records.append([label, len(good), "", "", ""])
            continue
        if variant == "low_min_daily":
            low = min(good.values()) if good else None
        else:
            low = _low_flow(good, n)
        peak = min(good, key=lambda d: (-good[d], d)) if good else None
        records.append([label, len(good), f"{fmean(good.values()):.3f}" if good else "",
                        "" if low is None else f"{low:.3f}", peak.isoformat() if peak else ""])
    return write_csv(HEADER, records)


def _number_matches(value: str, reference: str) -> bool:
    if reference == "":
        return value == ""
    try:
        return close(float(value), float(reference), TOLERANCE + 1e-9 * abs(float(reference)))
    except ValueError:
        return False


def _streamflow_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    first = text.splitlines()[0].strip() if text.strip() else ""
    out = rows(text)
    if first != ",".join(HEADER) or len(out) != len(reference):
        return [False]
    by_year = {row["water_year"].strip(): row for row in out}
    if set(by_year) != {row["water_year"] for row in reference}:
        return [False]
    results = []
    for ref in reference:
        got = {k: (v or "").strip() for k, v in by_year[ref["water_year"]].items() if k is not None}
        results += [got["approved_days"] == ref["approved_days"], _number_matches(got["mean_flow"], ref["mean_flow"]),
                    _number_matches(got["low_flow"], ref["low_flow"]), got["peak_date"] == ref["peak_date"]]
    return results


FAMILIES = (
    Family("streamflow", DOMAIN, _streamflow_build, _streamflow_check, _streamflow_solve,
           {trap: (LEVEL,) for trap in ("calendar_year", "label_by_start", "keep_provisional", "drop_estimated",
                                        "no_conversion", "ignore_completeness", "low_min_daily")},
           levels=(LEVEL,)),
)
