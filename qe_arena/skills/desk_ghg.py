"""natural-science desk family (level 4): annual greenhouse-gas means from a real NOAA GML in-situ monthly file.

See ``desk_science`` for what a desk family is. The input is a pinned NOAA Global Monitoring Laboratory file as
published: about 155 comment lines (variable attributes, fill values, the three-character QC flag), a column-name
line, then one whitespace-separated row per month with fill values (-999.99) for months without data and a count of
the days behind each monthly mean. Six series: CO2 at Barrow, Mauna Loa, American Samoa and the South Pole, CH4 at
Barrow and Mauna Loa.
"""

from __future__ import annotations

import gzip
import json
import random
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import NamedTuple

from . import Built, Family
from ._common import close, rows, write_csv

DOMAIN = "natural-science"
LEVEL = 4
DATA = Path(__file__).resolve().parent / "data" / "gml"
HEADER = ["year", "months", "mean", "amplitude", "growth"]
FILL = -999.99
TOLERANCE = 0.005 + 1e-6  # two decimal places, against the unrounded reference (any half-way rounding passes)
UNITS = {"co2": "ppm (micromol/mol)", "ch4": "ppb (nanomol/mol)"}
MIN_DAYS = (10, 12, 15, 18, 20, 22, 25)
MIN_MONTHS = (8, 9, 10, 11)


class Month(NamedTuple):
    year: int
    month: int
    value: float
    days: int
    flag: str


def _series() -> list[str]:
    return sorted(path.name.split("_monthly")[0] for path in DATA.glob("*_monthly.txt.gz"))


@lru_cache(maxsize=None)
def _snapshot(series: str) -> str:
    return gzip.decompress((DATA / f"{series}_monthly.txt.gz").read_bytes()).decode("utf-8")


@lru_cache(maxsize=8)
def _months(text: str) -> tuple[Month, ...]:
    """One record per data row; the first non-comment line names the columns."""
    lines = [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    names = lines[0]
    col = {name: names.index(name) for name in ("year", "month", "value", "nvalue", "qcflag")}
    return tuple(Month(int(c[col["year"]]), int(c[col["month"]]), float(c[col["value"]]), int(c[col["nvalue"]]),
                       c[col["qcflag"]]) for c in lines[1:])


def _qualifies(m: Month, min_days: int) -> bool:
    return m.value != FILL and m.flag[0] == "." and m.days >= min_days


def _binds(months: tuple[Month, ...], first: int, last: int, min_days: int, min_months: int) -> bool:
    """True when every rule changes the answer: a short month, an incomplete year with data, an incomplete year
    between complete ones (no fall-back), and a complete year before from_year (growth reaches outside)."""
    count: dict[int, int] = {}
    short = False
    for m in months:
        count[m.year] = count.get(m.year, 0) + _qualifies(m, min_days)
        short |= first <= m.year <= last and m.value != FILL and m.flag[0] == "." and 0 < m.days < min_days
    done = {year for year, n in count.items() if n >= min_months}
    window = range(first, last + 1)
    partial = any(0 < count.get(y, 0) < min_months for y in window)
    gap = any(y in done and y - 1 not in done and any(x in done for x in range(first, y - 1)) for y in window)
    return short and partial and gap and {first - 1, first} <= done


def _choose(months: tuple[Month, ...], rng: random.Random) -> dict[str, int]:
    """Request settings drawn until every rule binds on this series (the last draw stands after 40 tries, and the
    trap check then rejects the build if a rule does not bind)."""
    years = sorted({m.year for m in months})
    for _ in range(40):
        span = rng.randint(10, 20)
        first = rng.randint(years[0] + 1, years[-1] - span + 1)
        settings = {"from_year": first, "to_year": first + span - 1, "min_days": rng.choice(MIN_DAYS),
                    "min_months": rng.choice(MIN_MONTHS)}
        if _binds(months, *settings.values()):
            break
    return settings


def _ghg_build(level: int, rng: random.Random) -> Built:
    series = rng.choice(_series())
    gas, site = series.split("_")
    text = _snapshot(series)
    months = _months(text)
    request = _choose(months, rng)
    request = {"gas": gas, "site": site.upper(), **request}
    name = f"{gas}_{site}_surface-insitu_1_ccgg_MonthlyData.txt"
    files = {name: text, "request.json": json.dumps(request, indent=2) + "\n"}
    unit = UNITS[gas]
    statement = "\n".join([
        f"An atmospheric-chemistry group needs annual {gas.upper()} statistics for the {site.upper()} observatory "
        f"from {name}, NOAA GML's in-situ monthly means as published. Lines starting with # are comments (they "
        "document every column, the fill values and the quality flag); the first other line names the columns, "
        f"and each row after it is one month, whitespace-separated. value is the monthly mean in {unit}; nvalue "
        "is the number of days behind it; qcflag is a three-character flag. request.json gives the settings.",
        "",
        "Rules:",
        "- A month qualifies only if its value is not the fill value -999.99, the first character of its qcflag "
        "is a period (.), and its nvalue is at least min_days. Other months are left out of every statistic.",
        "- Work by calendar year, using the year column. A year is complete when it has at least min_months "
        "qualifying months.",
        "- months is the number of qualifying months in the year. For a complete year, mean is the mean of its "
        "qualifying monthly values and amplitude is the highest minus the lowest of them; for an incomplete year "
        "leave both empty.",
        "- growth is the year's mean minus the previous calendar year's mean, worked out by the same rules even "
        "when the previous year is before from_year. Leave it empty unless both years are complete; never fall "
        "back to an earlier year.",
        "",
        "Write annual_means.csv with exactly this header line: " + ",".join(HEADER) + ", then one row for every "
        "year from from_year to to_year, both included, that has at least one row in the file, in ascending "
        "year order. Round mean, amplitude and growth to 2 decimal places (compute growth from unrounded means). "
        "The file is too long to print in full, so process it with a script.",
    ])
    built = Built(files=files, statement=statement, output="annual_means.csv", reference=None)
    built.reference = rows(_table(files, None, 6))
    return built


def _ghg_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    return _table(files, variant, 2)


def _table(files: dict[str, str], variant: str | None, digits: int) -> str:
    request = json.loads(files["request.json"])
    data = next(text for name, text in files.items() if name.endswith(".txt"))
    min_days, need = request["min_days"], request["min_months"]
    first, last = request["from_year"], request["to_year"]
    if variant == "to_year_exclusive":
        last -= 1
    by_year: dict[int, list[float]] = {}
    for m in _months(data):
        if variant == "average_everything":
            good = True
        elif variant == "ignore_qcflag":
            good = m.value != FILL and m.days >= min_days
        elif variant == "ignore_min_days":
            good = m.value != FILL and m.flag[0] == "."
        else:
            good = _qualifies(m, min_days)
        by_year.setdefault(m.year, [])
        if good:
            by_year[m.year].append(m.value)
    complete = {y for y, values in by_year.items()
                if values and (len(values) >= need or variant == "ignore_completeness")}
    means = {y: fmean(by_year[y]) for y in complete}
    records = []
    previous = None  # the latest complete year seen in the window, for the wrong fall-back variant
    for year in sorted(y for y in by_year if first <= y <= last):
        values = by_year[year]
        if year not in complete:
            records.append([year, len(values), "", "", ""])
            continue
        if variant == "growth_skip_gap":
            before = previous if previous is not None else (year - 1 if year - 1 in complete else None)
        elif variant == "growth_window_only":
            before = year - 1 if year - 1 in complete and year > first else None
        else:
            before = year - 1 if year - 1 in complete else None
        growth = "" if before is None else f"{means[year] - means[before]:.{digits}f}"
        records.append([year, len(values), f"{means[year]:.{digits}f}", f"{max(values) - min(values):.{digits}f}",
                        growth])
        previous = year
    return write_csv(HEADER, records)


def _number_matches(value: str, reference: str) -> bool:
    if reference == "":
        return value == ""
    try:
        return close(float(value), float(reference), TOLERANCE)
    except ValueError:
        return False


def _ghg_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    first = text.splitlines()[0].strip() if text.strip() else ""
    out = rows(text)
    if first != ",".join(HEADER) or len(out) != len(reference):
        return [False]
    results = []
    for got, ref in zip(out, reference):
        got = {k: (v or "").strip() for k, v in got.items() if k is not None}
        results += [got["year"] == ref["year"], got["months"] == ref["months"],
                    *(_number_matches(got[key], ref[key]) for key in ("mean", "amplitude", "growth"))]
    return results


# ignore_qcflag is a solver variant too, but in these files every month whose flag rejects a real value also has
# fewer than 10 days, so the min_days rule already drops it and the variant never changes the answer.
TRAPS = ("average_everything", "ignore_min_days", "ignore_completeness", "growth_skip_gap", "growth_window_only",
         "to_year_exclusive")
FAMILIES = (
    Family("ghg", DOMAIN, _ghg_build, _ghg_check, _ghg_solve, {trap: (LEVEL,) for trap in TRAPS}, levels=(LEVEL,)),
)
