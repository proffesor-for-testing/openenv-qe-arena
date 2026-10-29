"""finance-economics desk family (level 4): yield-curve inversion spells from real US Treasury par yield files.

See ``desk_science`` for what a desk family is. The input is two or three of Treasury's yearly CSV downloads as
published: newest row first, MM/DD/YYYY dates, a column set that changes between years (4 Mo from 2022, 1.5 Month
from 2025) and blank cells where a tenor was not yet quoted.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import json
import random
from functools import lru_cache
from pathlib import Path

from . import Built, Family
from ._common import rows, write_csv

DOMAIN = "finance-economics"
LEVEL = 4
DATA = Path(__file__).resolve().parent / "data" / "treasury"
YEARS = tuple(range(2018, 2027))
HEADER = ["start", "end", "trading_days", "min_spread_bp", "min_spread_date"]
PAIRS = (("10 Yr", "2 Yr"), ("10 Yr", "3 Mo"), ("30 Yr", "3 Mo"), ("5 Yr", "3 Mo"), ("2 Yr", "3 Mo"),
         ("1 Yr", "4 Mo"), ("6 Mo", "2 Mo"), ("30 Yr", "1 Mo"), ("7 Yr", "1 Yr"), ("3 Mo", "1.5 Month"))


@lru_cache(maxsize=None)
def _year(year: int) -> str:
    return gzip.decompress((DATA / f"par_yield_curve_{year}.csv.gz").read_bytes()).decode("utf-8")


def _file(year: int) -> str:
    return f"daily_treasury_par_yield_curve_{year}.csv"


@lru_cache(maxsize=32)
def _days(text: str) -> tuple[dict[str, str], ...]:
    return tuple(csv.DictReader(io.StringIO(text)))


def _yields_build(level: int, rng: random.Random) -> Built:
    first = rng.choice(YEARS[:-1])
    years = [y for y in range(first, first + rng.choice((2, 3))) if y in YEARS]
    long, short = rng.choice(PAIRS)
    request = {"long_tenor": long, "short_tenor": short, "min_spell_days": rng.choice((2, 3, 5, 10))}
    files = {_file(y): _year(y) for y in years}
    files["request.json"] = json.dumps(request, indent=2) + "\n"
    statement = "\n".join([
        f"The rates desk wants every inversion spell of the {long} minus {short} spread in "
        f"{', '.join(_file(y) for y in years)}: Treasury's daily par yield curve downloads, one file per year, "
        "yields in percent. request.json names the two tenors and the shortest spell to report. Read the files "
        "as they are: rows run newest first, dates are MM/DD/YYYY, and the columns differ between years.",
        "",
        "Rules:",
        "- Put every trading day from all files in date order. The spread on a day is the long tenor's yield minus "
        "the short tenor's, in basis points (1 percentage point = 100 bp).",
        "- A day where either tenor is blank, or whose file has no column for it, is skipped: it is not counted and "
        "it does not break a spell.",
        "- A spell is a run of consecutive counted days whose spread is strictly below zero (a spread of exactly 0 "
        "is not inverted). Report only spells of at least min_spell_days counted days; a spell may cross from one "
        "file into the next.",
        "- For each spell give its first and last counted day (YYYY-MM-DD), trading_days (the number of counted "
        "days in it), min_spread_bp (its most negative spread, a whole number of bp) and min_spread_date (the "
        "earliest day with that spread).",
        "",
        "Write inversion_spells.csv with exactly this header line: " + ",".join(HEADER) + ", then one row per "
        "spell in date order. Write only the header when there is no qualifying spell. Use a script; the files are "
        "too long to read by eye.",
    ])
    built = Built(files=files, statement=statement, output="inversion_spells.csv", reference=None)
    built.reference = rows(_yields_solve(files, level, None))
    return built


def _yields_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    long, short = request["long_tenor"], request["short_tenor"]
    if variant == "reversed_spread":
        long, short = short, long
    days: list[tuple[dt.date, int | None]] = []
    for name in sorted(n for n in files if n.endswith(".csv")):
        for day in _days(files[name]):
            date = dt.datetime.strptime(day["Date"], "%m/%d/%Y").date()
            a, b = (day.get(long) or "").strip(), (day.get(short) or "").strip()
            days.append((date, round((float(a) - float(b)) * 100) if a and b else None))
    if variant != "file_order":
        days.sort()
    spells, run = [], []
    for date, spread in days + [(None, None)]:
        if spread is None and date is not None and variant != "break_on_missing":
            continue
        inverted = spread is not None and (spread <= 0 if variant == "le_zero" else spread < 0)
        if inverted:
            run.append((date, spread))
            continue
        if run and (len(run) >= request["min_spell_days"] or variant == "ignore_min_spell"):
            low = min(s for _, s in run)
            length = (run[-1][0] - run[0][0]).days + 1 if variant == "calendar_days" else len(run)
            spells.append([run[0][0].isoformat(), run[-1][0].isoformat(), length,
                           low / 100 if variant == "percent_units" else low,
                           next(d for d, s in run if s == low).isoformat()])
        run = []
    return write_csv(HEADER, spells)


def _yields_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[0].strip() != ",".join(HEADER):
        return [False]
    out = rows(text)
    if len(out) != len(reference):
        return [False]
    results = []
    for got, ref in zip(out, reference):
        got = {k: (v or "").strip() for k, v in got.items() if k is not None}
        try:
            bp = float(got["min_spread_bp"]) == float(ref["min_spread_bp"])
        except ValueError:
            bp = False
        results += [got["start"] == ref["start"], got["end"] == ref["end"],
                    got["trading_days"] == ref["trading_days"], bp, got["min_spread_date"] == ref["min_spread_date"]]
    return results or [True]


FAMILIES = (
    Family("yields", DOMAIN, _yields_build, _yields_check, _yields_solve,
           # break_on_missing is a solver variant too, but no blank cell in the pinned files falls inside a spell, so
           # it never changes the answer and cannot be required to fail.
           {trap: (LEVEL,) for trap in ("reversed_spread", "file_order", "le_zero", "ignore_min_spell",
                                        "calendar_days", "percent_units")},
           levels=(LEVEL,)),
)
