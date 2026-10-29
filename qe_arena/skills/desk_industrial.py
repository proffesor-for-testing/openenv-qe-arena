"""industrial-physical-systems desk family (level 4): mainshock declustering of a real USGS earthquake catalogue.

See ``desk_science`` for what a desk family is. The input is a pinned ComCat slice exactly as the FDSN event service
returns it (newest first, CSV-quoted place names, a mix of reviewed and automatic solutions, several magnitude
scales, quarry blasts and explosions), 600-1,300 events, so the model has to script it.
"""

from __future__ import annotations

import bisect
import csv
import datetime as dt
import gzip
import io
import json
import math
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from . import Built, Family
from ._common import rows, write_csv

DOMAIN = "industrial-physical-systems"
LEVEL = 4
DATA = Path(__file__).resolve().parent / "data" / "quakes"
FILE = "catalogue.csv"
HEADER = ["id", "time", "mag", "magType", "aftershocks"]
EARTH_KM = 6371.0
KM_PER_DEGREE = math.pi * EARTH_KM / 180
# Gardner and Knopoff (1974) windows from M2.5 up, with two rows added below for the small events of these slices.
GK = ((0.0, 10.0, 2.0), (2.0, 15.0, 4.0), (2.5, 19.5, 6.0), (3.0, 22.5, 11.5), (3.5, 26.0, 22.0), (4.0, 30.0, 42.0),
      (4.5, 35.0, 83.0), (5.0, 40.0, 155.0), (5.5, 47.0, 290.0))
KM_SCALES = (1.0, 1.25, 1.5)
DAY_SCALES = (1.0, 2.0)


class Event(NamedTuple):
    time: str
    t: float  # seconds since the epoch
    lat: float
    lon: float
    depth: float
    mag: float
    mag_text: str  # as written in the file (1.6, 1.66, 2.13)
    mag_type: str
    id: str
    kind: str
    status: str


def _slices() -> list[str]:
    return sorted(path.name.split(".")[0] for path in DATA.glob("*.csv.gz"))


# The columns the task needs, in ComCat's own order. Serving all 22 made one 40-line peek cost 8 KiB of output, which
# spent the whole completion budget on the first step in all 16 screen episodes (2026-10-28 adjustment).
SERVED_COLUMNS = ("time", "latitude", "longitude", "depth", "mag", "magType", "id", "place", "type", "status")


@lru_cache(maxsize=None)
def _snapshot(name: str) -> str:
    """The pinned slice as published, cut to SERVED_COLUMNS; rows, order, quoting and values are unchanged."""
    raw = gzip.decompress((DATA / f"{name}.csv.gz").read_bytes()).decode("utf-8")
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    reader = csv.DictReader(io.StringIO(raw))
    writer.writerow(SERVED_COLUMNS)
    for row in reader:
        writer.writerow([row[c] for c in SERVED_COLUMNS])
    return out.getvalue()


def _seconds(text: str) -> float:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


@lru_cache(maxsize=8)
def _events(text: str) -> tuple[Event, ...]:
    """Every catalogue row in file order; cached because each trap check of a build re-reads the same text."""
    return tuple(Event(r["time"], _seconds(r["time"]), float(r["latitude"]), float(r["longitude"]),
                       float(r["depth"] or 0), float(r["mag"]), r["mag"], r["magType"], r["id"], r["type"], r["status"])
                 for r in csv.DictReader(io.StringIO(text)))


def _haversine(a: Event, b: Event) -> float:
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(h)))


def _flat(a: Event, b: Event) -> float:
    return KM_PER_DEGREE * math.hypot(b.lat - a.lat, b.lon - a.lon)


def _hypocentral(a: Event, b: Event) -> float:
    return math.hypot(_haversine(a, b), b.depth - a.depth)


def _window(table: list[list[float]], mag: float) -> tuple[float, float]:
    row = [r for r in table if r[0] <= mag][-1]
    return row[1], row[2]


def _quakes_build(level: int, rng: random.Random) -> Built:
    name = rng.choice(_slices())
    text = _snapshot(name)
    events = sorted(_events(text), key=lambda e: (e.t, e.id))
    good = [e for e in events if e.kind == "earthquake" and e.status == "reviewed"]
    # The window starts and ends on real qualifying events and the magnitude floor is a magnitude that occurs, so the
    # inclusive-start, exclusive-end and at-or-above rules change the answer instead of being decorative.
    days = rng.randint(30, 70)
    first = rng.choice([e for e in good if e.t <= good[-1].t - days * 86400])
    last = next(e for e in good if e.t >= first.t + days * 86400)
    inside = [e for e in good if first.t <= e.t < last.t]
    floor = min(first.mag, last.mag, 2.4)
    min_mag = rng.choice(sorted({e.mag for e in inside if 1.5 <= e.mag <= floor}))
    counts = Counter(e.mag_type for e in inside if e.mag >= min_mag)
    dominant = counts.most_common(1)[0][0]
    keep = {dominant, first.mag_type, last.mag_type}
    others = sorted(set(counts) - keep)
    drop = {rng.choice(others)} if others else set()
    mag_types = sorted(({e.mag_type for e in events} - drop) | keep)
    km, day = rng.choice(KM_SCALES), rng.choice(DAY_SCALES)
    table = [[m, round(d * km, 1), round(t * day, 1)] for m, d, t in GK]
    request = {"start": first.time, "end": last.time, "min_magnitude": min_mag, "mag_types": mag_types,
               "windows": [dict(zip(("min_mag", "distance_km", "days"), row)) for row in table]}
    files = {FILE: text, "request.json": json.dumps(request, indent=2) + "\n"}
    statement = "\n".join([
        f"A seismic-hazard team needs the mainshocks of a USGS earthquake catalogue. {FILE} is a ComCat export "
        "(FDSN event CSV, newest event first; place names are CSV-quoted and contain commas); times are UTC, ISO "
        "8601. request.json gives the time window, the magnitude floor, the magnitude types to use and a "
        "declustering window table.",
        "",
        "Rules:",
        "- Use an event only if its type is earthquake, its status is reviewed (automatic solutions are not), "
        "its magType is listed in mag_types, its mag is at or above min_magnitude, and its time is at or after "
        "start and strictly before end.",
        "- Decluster the selected events in time order, oldest first. An event is an aftershock when some "
        "mainshock earlier in that order has a magnitude at least the event's own, occurred at most D days "
        "before it (1 day = 86400 s) and lies at most L km from it, where D and L come from the mainshock's row "
        "of windows: the last row whose min_mag is at or below the mainshock's mag. Any other event is a "
        "mainshock. Count each aftershock once, for its qualifying mainshock of largest magnitude (the earliest "
        "on a tie).",
        "- L is the great-circle distance between epicentres (haversine, Earth radius 6371.0 km); ignore depth.",
        "",
        "Write mainshocks.csv with exactly this header line: " + ",".join(HEADER) + ", then one row per mainshock "
        "in time order, oldest first. Copy id, time, mag and magType as they appear in the catalogue; aftershocks "
        "is the number of aftershocks counted for that mainshock. The catalogue is too long to print, so process "
        "it with a script.",
    ])
    built = Built(files=files, statement=statement, output="mainshocks.csv", reference=None)
    built.reference = rows(_quakes_solve(files, level, None))
    return built


def _quakes_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    start, end = _seconds(request["start"]), _seconds(request["end"])
    floor, table = request["min_magnitude"], [[w["min_mag"], w["distance_km"], w["days"]] for w in request["windows"]]
    selected = []
    for e in _events(files[FILE]):
        if e.kind != "earthquake" and variant != "keep_non_earthquakes":
            continue
        if e.status != "reviewed" and variant != "keep_automatic":
            continue
        if e.mag_type not in request["mag_types"] and variant != "all_mag_types":
            continue
        if e.mag < floor or (variant == "mag_floor_exclusive" and e.mag == floor):
            continue
        if not (start < e.t if variant == "start_exclusive" else start <= e.t):
            continue
        if not (e.t <= end if variant == "end_inclusive" else e.t < end):
            continue
        selected.append(e)
    if variant != "newest_first":  # the file lists newest first; the rule says oldest first
        selected.sort(key=lambda e: (e.t, e.id))
    distance = _flat if variant == "flat_earth" else _hypocentral if variant == "hypocentral" else _haversine
    longest = max(row[2] for row in table) * 86400
    mains: list[Event] = []
    times: list[float] = []  # mains' times, ascending unless the order is wrong (newest_first)
    counts: dict[str, int] = {}
    for e in selected:
        best = None
        for m in mains if variant == "newest_first" else mains[bisect.bisect_left(times, e.t - longest):]:
            if m.mag < e.mag and variant != "larger_aftershocks":
                continue
            km, days = _window(table, e.mag if variant == "window_from_aftershock" else m.mag)
            if not 0 <= e.t - m.t <= days * 86400 or distance(m, e) > km:
                continue
            if best is None or (m.mag > best.mag and variant != "assign_earliest"):
                best = m
        if best is None:
            mains.append(e)
            times.append(e.t)
            counts[e.id] = 0
        else:
            counts[best.id] += 1
    if variant == "by_magnitude":
        mains.sort(key=lambda m: (-m.mag, m.t))
    elif variant == "newest_first":
        mains.sort(key=lambda m: (m.t, m.id))
    return write_csv(HEADER, [[m.id, m.time, m.mag_text, m.mag_type, counts[m.id]] for m in mains])


def _quakes_check(reference: list[dict[str, str]], text: str) -> list[bool]:
    first = text.splitlines()[0].strip() if text.strip() else ""
    out = rows(text)
    if first != ",".join(HEADER) or len(out) != len(reference):
        return [False]
    results = []
    for got, ref in zip(out, reference):
        got = {k: (v or "").strip() for k, v in got.items() if k is not None}
        try:
            mag = math.isfinite(float(got["mag"])) and abs(float(got["mag"]) - float(ref["mag"])) < 1e-9
        except ValueError:
            mag = False
        results += [got["id"] == ref["id"], got["time"] == ref["time"], mag, got["magType"] == ref["magType"],
                    got["aftershocks"] == ref["aftershocks"]]
    return results


TRAPS = ("keep_automatic", "keep_non_earthquakes", "all_mag_types", "mag_floor_exclusive", "start_exclusive",
         "end_inclusive", "flat_earth", "newest_first", "by_magnitude", "larger_aftershocks", "window_from_aftershock",
         "assign_earliest")
# hypocentral (distance including depth) is a solver variant too, but on these slices it changes the answer on only
# about 40% of requests, so it is not required to fail on every seed. flat_earth changes it on about 80%; the build
# retries cover the rest.
FAMILIES = (
    Family("quakes", DOMAIN, _quakes_build, _quakes_check, _quakes_solve, {trap: (LEVEL,) for trap in TRAPS},
           levels=(LEVEL,)),
)
