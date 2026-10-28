"""office-white-collar desk family (level 4): meeting hours and double bookings from a synthetic ICS export.

See ``desk_science`` for what a desk family is. The calendar is synthetic (a real export would carry people's
schedules), but it is shaped like one from a groupware server: VTIMEZONE blocks, folded ATTENDEE lines with quoted
parameters, recurring series (weekly and daily, INTERVAL, BYDAY, COUNT or UNTIL) with EXDATEs in either local or UTC
form, all-day events, cancelled events and declined invitations. The report range always spans the weeks in which
the US and the EU have switched daylight-saving time on different dates, and a planted pair of series in a US and
an EU zone only collides outside those weeks, so a fixed-offset reading changes the answer.

Python's standard library only: zoneinfo for the zones, and a small RFC 5545 subset reader below.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import random
import re
from collections.abc import Callable, Iterator
from zoneinfo import ZoneInfo

from . import Built, Family
from ._common import rows, write_csv

DOMAIN = "office-white-collar"
LEVEL = 4
FILE = "calendar.ics"
OUTPUT = "meeting_hours.csv"
HEADER = ["person", "week", "meeting_hours", "conflicts"]
UTC = dt.timezone.utc
US_ZONES = ("America/New_York", "America/Chicago", "America/Los_Angeles")
EU_ZONES = ("Europe/London", "Europe/Berlin", "Europe/Paris")
EVENT_ZONES = (*US_ZONES, *EU_ZONES, "Asia/Tokyo", "UTC")
REPORT_ZONES = ("Europe/Berlin", "America/New_York", "Europe/London", "America/Chicago", "Asia/Tokyo")
YEARS = (2025, 2026, 2027)
DAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
NAMES = (("Ana", "Ruiz"), ("Ben", "Okafor"), ("Chloe", "Martin"), ("Dev", "Patel"), ("Elif", "Kaya"),
         ("Femi", "Adeyemi"), ("Grace", "Liu"), ("Hugo", "Berg"), ("Ines", "Costa"), ("Jonas", "Weber"),
         ("Kenji", "Sato"), ("Lena", "Novak"), ("Mara", "Silva"), ("Nils", "Holm"))
COMPANIES = ("northwind.example", "contoso.example", "fabrikam.example")
SUMMARIES = ("Standup", "1:1", "Design review", "Sprint planning", "Customer call", "Budget sync", "Hiring panel",
             "Vendor demo", "Retro", "All hands", "Roadmap review", "Ops handover", "Quarterly close",
             "Interview debrief", "Security review", "Pricing workshop")
ALL_DAY = ("Offsite", "Public holiday", "Conference", "Training day", "Out of office", "Release freeze")
LOCATIONS = ("Room 4.12", "Zoom", "Teams", "Atrium", "Board room", "Room 2.03", "Lisbon office")
MINUTES = (15, 20, 25, 30, 45, 50, 55, 60, 75, 90, 120)
PARTSTATS = ("ACCEPTED",) * 12 + ("TENTATIVE",) * 2 + ("NEEDS-ACTION",) * 3 + ("DECLINED",) * 2 + (None,)
STATUSES = ("CONFIRMED",) * 14 + ("TENTATIVE",) * 3 + (None,) * 2 + ("CANCELLED",)
VTIMEZONES = {
    "America/New_York": ("-0500", "-0400", "EST", "EDT", "0308T020000", "2SU", "1101T020000", "1SU", 11),
    "America/Chicago": ("-0600", "-0500", "CST", "CDT", "0308T020000", "2SU", "1101T020000", "1SU", 11),
    "America/Los_Angeles": ("-0800", "-0700", "PST", "PDT", "0308T020000", "2SU", "1101T020000", "1SU", 11),
    "Europe/London": ("+0000", "+0100", "GMT", "BST", "0329T010000", "-1SU", "1025T020000", "-1SU", 10),
    "Europe/Berlin": ("+0100", "+0200", "CET", "CEST", "0329T020000", "-1SU", "1025T030000", "-1SU", 10),
    "Europe/Paris": ("+0100", "+0200", "CET", "CEST", "0329T020000", "-1SU", "1025T030000", "-1SU", 10),
}
DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")
FIELD = re.compile(r"^(\d{8})(?:T(\d{6})(Z?))?$")

Converter = Callable[[dt.datetime, str], dt.datetime]


# ---------------------------------------------------------------- reading (shared by the solver and the generator)

def _unfold(text: str, variant: str | None = None) -> list[str]:
    """Content lines with RFC 5545 folding undone (a line starting with a space or tab continues the last one)."""
    out: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line[:1] in (" ", "\t"):
            if out and variant != "no_unfold":
                out[-1] += line[1:]
            continue
        out.append(line)
    return out


def _split(text: str, sep: str) -> list[str]:
    """``text`` split on ``sep`` outside double quotes."""
    parts, quoted, start = [], False, 0
    for i, ch in enumerate(text):
        if ch == '"':
            quoted = not quoted
        elif ch == sep and not quoted:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def _prop(line: str) -> tuple[str, dict[str, str], str] | None:
    """(NAME, {PARAM: value}, value) of a content line; the value starts after the first colon outside quotes."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == ":" and not quoted:
            name, *params = _split(line[:i], ";")
            pairs = (p.split("=", 1) for p in params if "=" in p)
            return name.upper(), {k.upper(): v.strip('"') for k, v in pairs}, line[i + 1:]
    return None


def _when(value: str, params: dict[str, str]) -> tuple[dt.datetime, str] | dt.date:
    """A DATE (all-day) or a (naive wall time, zone) pair; a trailing Z means UTC."""
    m = FIELD.match(value.strip())
    if not m:
        raise ValueError(f"bad date-time {value!r}")
    day = dt.datetime.strptime(m.group(1), "%Y%m%d")
    if m.group(2) is None or params.get("VALUE") == "DATE":
        return day.date()
    wall = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return wall, "UTC" if m.group(3) else params.get("TZID", "UTC")


def _duration(value: str) -> dt.timedelta:
    m = DURATION.match(value.strip())
    if not m:
        raise ValueError(f"bad duration {value!r}")
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return dt.timedelta(days=d, hours=h, minutes=mi, seconds=s)


def _rule(value: str) -> dict:
    parts = dict(p.split("=", 1) for p in value.split(";") if "=" in p)
    rule = {"FREQ": parts["FREQ"], "INTERVAL": int(parts.get("INTERVAL", 1)),
            "BYDAY": [DAYS.index(d[-2:]) for d in parts["BYDAY"].split(",")] if "BYDAY" in parts else [],
            "COUNT": int(parts["COUNT"]) if "COUNT" in parts else None, "UNTIL": None}
    if "UNTIL" in parts:
        rule["UNTIL"] = _when(parts["UNTIL"], {})
    return rule


def _events(text: str, variant: str | None) -> list[dict]:
    events, current = [], None
    for line in _unfold(text, variant):
        if line == "BEGIN:VEVENT":
            current = {"exdates": [], "people": {}, "status": "", "rrule": None, "length": None}
            continue
        if line == "END:VEVENT" and current is not None:
            events.append(current)
            current = None
            continue
        if current is None or (parsed := _prop(line)) is None:
            continue
        name, params, value = parsed
        if name == "DTSTART":
            current["start"] = _when(value, params)
        elif name == "DTEND":
            current["end"] = _when(value, params)
        elif name == "DURATION":
            current["length"] = _duration(value)
        elif name == "RRULE":
            current["rrule"] = _rule(value)
        elif name == "EXDATE":
            current["exdates"] += [_when(v, params) for v in value.split(",") if v.strip()]
        elif name == "STATUS":
            current["status"] = value.strip().upper()
        elif name in ("ORGANIZER", "ATTENDEE"):
            email = value.strip()
            if email[:7].lower() == "mailto:":
                email = email[7:]
            key = email if variant == "case_sensitive" else email.lower()
            declined = name == "ATTENDEE" and params.get("PARTSTAT", "").upper() == "DECLINED"
            current["people"][key] = current["people"].get(key, False) or not declined or variant == "count_declined"
    return events


def _starts(start: dt.datetime, rule: dict | None, to_utc: Callable[[dt.datetime], dt.datetime],
            excluded: set[dt.datetime], horizon: dt.datetime, variant: str | None = None) -> Iterator[dt.datetime]:
    """Wall-clock start times of a series, EXDATEs removed, up to ``horizon`` (UTC)."""
    if rule is None:
        if to_utc(start) not in excluded:
            yield start
        return
    step = rule["INTERVAL"]
    if rule["FREQ"] == "DAILY":
        candidates: Iterator[dt.datetime] = (start + dt.timedelta(days=step * k) for k in itertools.count())
    elif rule["FREQ"] == "WEEKLY":
        monday = start.date() - dt.timedelta(days=start.weekday())
        days = sorted(set(rule["BYDAY"])) or [start.weekday()]
        candidates = (dt.datetime.combine(monday + dt.timedelta(weeks=step * k, days=d), start.time())
                      for k in itertools.count() for d in days)
    else:
        raise ValueError(f"unsupported FREQ {rule['FREQ']}")
    seen = 0
    for when in candidates:
        if when < start:
            continue
        instant = to_utc(when)
        until = rule["UNTIL"]
        if until is not None and (instant >= until if variant == "until_exclusive" else instant > until):
            return
        if rule["COUNT"] is not None and seen >= rule["COUNT"]:
            return
        if instant > horizon:
            return
        if instant in excluded:
            if variant != "count_skips_exdate":
                seen += 1
            continue
        seen += 1
        yield when


def _converter(variant: str | None, report: ZoneInfo, event_zone: str, first: dt.datetime) -> Converter:
    fixed = first.replace(tzinfo=ZoneInfo(event_zone)).utcoffset()

    def to_utc(wall: dt.datetime, zone: str) -> dt.datetime:
        if variant == "ignore_tz":
            return wall.replace(tzinfo=report).astimezone(UTC)
        if variant == "fixed_offset" and zone == event_zone:
            return wall.replace(tzinfo=dt.timezone(fixed)).astimezone(UTC)
        return wall.replace(tzinfo=ZoneInfo(zone)).astimezone(UTC)
    return to_utc


def _week(instant: dt.datetime, report: ZoneInfo) -> str:
    year, week, _ = instant.astimezone(report).isocalendar()
    return f"{year}-W{week:02d}"


def _calendar_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    report = ZoneInfo(request["time_zone"])
    weeks = []
    monday = dt.date.fromisoformat(request["week_from"] + "-1")
    while (label := f"{monday.isocalendar()[0]}-W{monday.isocalendar()[1]:02d}") <= request["week_to"]:
        weeks.append(label)
        monday += dt.timedelta(weeks=1)
    horizon = dt.datetime.combine(monday + dt.timedelta(days=2), dt.time(), report).astimezone(UTC)
    people = {(p if variant == "case_sensitive" else p.lower()): p for p in request["people"]}
    spans: dict[str, list[tuple[dt.datetime, dt.datetime]]] = {p: [] for p in request["people"]}
    for ev in _events(files[FILE], variant):
        if ev["status"] == "CANCELLED" and variant != "keep_cancelled":
            continue
        if "start" not in ev:
            continue
        attendees = [people[k] for k, attends in ev["people"].items() if attends and k in people]
        if not attendees:
            continue
        if isinstance(ev["start"], dt.date) and not isinstance(ev["start"], dt.datetime):
            if variant != "allday_24h":
                continue
            end = ev.get("end")
            days = (end - ev["start"]).days if isinstance(end, dt.date) else 1
            start, zone, length = dt.datetime.combine(ev["start"], dt.time()), request["time_zone"], \
                dt.timedelta(days=days)
        else:
            start, zone = ev["start"]
            if ev["length"] is not None:
                length = ev["length"]
            else:
                end_wall, end_zone = ev["end"]
                length = end_wall.replace(tzinfo=ZoneInfo(end_zone)) - start.replace(tzinfo=ZoneInfo(zone))
        to_utc = _converter(variant, report, zone, start)
        excluded = set() if variant == "ignore_exdate" else {
            to_utc(*x) for x in ev["exdates"] if isinstance(x, tuple)}
        rule = ev["rrule"]
        if rule is not None and rule["UNTIL"] is not None and isinstance(rule["UNTIL"], tuple):
            rule = dict(rule, UNTIL=to_utc(*rule["UNTIL"]))
        for wall in _starts(start, rule, lambda w: to_utc(w, zone), excluded, horizon, variant):
            begin = to_utc(wall, zone)
            for person in attendees:
                spans[person].append((begin, begin + length))
    out = []
    for person in request["people"]:
        hours = dict.fromkeys(weeks, 0.0)
        clashes = dict.fromkeys(weeks, 0)
        occ = sorted(spans[person])
        for i, (begin, end) in enumerate(occ):
            week = _week(begin, report)
            if week in hours:
                hours[week] += (end - begin).total_seconds() / 3600
                clashes[week] += sum(1 for other, _ in occ[i + 1:] if other < end)
        out += [[person, w, f"{hours[w]:.2f}", clashes[w]] for w in weeks]
    return write_csv(HEADER, out)


def _calendar_check(reference: list[dict[str, str]], text: str) -> list[bool]:
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
            hours = abs(float(got["meeting_hours"]) - float(ref["meeting_hours"])) <= 0.005 + 1e-9
        except ValueError:
            hours = False
        results += [got["person"] == ref["person"], got["week"] == ref["week"], hours,
                    got["conflicts"] == ref["conflicts"]]
    return results


# ---------------------------------------------------------------- generation

def _utc(wall: dt.datetime, zone: str) -> dt.datetime:
    return wall.replace(tzinfo=ZoneInfo(zone)).astimezone(UTC)


def _change(zone: str, year: int, month: int) -> dt.date:
    """The day in ``month`` on which ``zone`` changes its UTC offset."""
    tz = ZoneInfo(zone)
    day = dt.date(year, month, 1) - dt.timedelta(days=1)
    for _ in range(31):
        nxt = day + dt.timedelta(days=1)
        if dt.datetime.combine(nxt, dt.time(12), tz).utcoffset() != dt.datetime.combine(day, dt.time(12), tz).utcoffset():
            return nxt
        day = nxt
    raise ValueError(f"{zone} does not change offset in {year}-{month:02d}")


def _fold(line: str) -> list[str]:
    out = [line[:75]]
    line = line[75:]
    while line:
        out.append(" " + line[:74])
        line = line[74:]
    return out


def _vtimezone(zone: str) -> list[str]:
    if zone == "Asia/Tokyo":
        return ["BEGIN:VTIMEZONE", "TZID:Asia/Tokyo", "BEGIN:STANDARD", "TZOFFSETFROM:+0900", "TZOFFSETTO:+0900",
                "TZNAME:JST", "DTSTART:19700101T000000", "END:STANDARD", "END:VTIMEZONE"]
    std, dst, std_name, dst_name, on, on_day, off, off_day, off_month = VTIMEZONES[zone]
    return ["BEGIN:VTIMEZONE", f"TZID:{zone}", "BEGIN:DAYLIGHT", f"TZOFFSETFROM:{std}", f"TZOFFSETTO:{dst}",
            f"TZNAME:{dst_name}", f"DTSTART:1970{on}", f"RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY={on_day}", "END:DAYLIGHT",
            "BEGIN:STANDARD", f"TZOFFSETFROM:{dst}", f"TZOFFSETTO:{std}", f"TZNAME:{std_name}", f"DTSTART:1970{off}",
            f"RRULE:FREQ=YEARLY;BYMONTH={off_month};BYDAY={off_day}", "END:STANDARD", "END:VTIMEZONE"]


class _Gen:
    """Builds the event specs of one calendar and serialises them."""

    def __init__(self, rng: random.Random, people: list[tuple[str, str, str]], start: dt.date, end: dt.date):
        self.rng, self.people, self.start, self.end = rng, people, start, end
        self.events: list[dict] = []

    def email(self, person: tuple[str, str, str], shout: bool = False) -> str:
        first, last, domain = person
        if shout:
            return f"{first}.{last}@{domain[0].upper()}{domain[1:]}"
        return f"{first}.{last}@{domain}".lower()

    def wall(self, lo: dt.date | None = None, hi: dt.date | None = None, hours: tuple[int, int] = (8, 17)) -> dt.datetime:
        lo, hi = lo or self.start, hi or self.end
        while True:
            day = lo + dt.timedelta(days=self.rng.randrange((hi - lo).days))
            if day.weekday() < 5:
                return dt.datetime.combine(day, dt.time(self.rng.randint(*hours), self.rng.randrange(0, 60, 5)))

    def add(self, *, start, zone="UTC", minutes=30, rule=None, exdates=(), status="CONFIRMED", organizer=None,
            attendees=(), allday_days=0, summary=None) -> dict:
        rng = self.rng
        ev = {"uid": f"{rng.getrandbits(64):016x}-{len(self.events)}@groupware", "start": start, "zone": zone,
              "minutes": minutes, "rule": rule, "exdates": list(exdates), "status": status,
              "organizer": organizer or self.email(rng.choice(self.people)), "attendees": list(attendees),
              "allday_days": allday_days, "duration_prop": rng.random() < 0.3,
              "exdate_utc": zone == "UTC" or rng.random() < 0.35,
              "summary": summary or rng.choice(ALL_DAY if allday_days else SUMMARIES)}
        self.events.append(ev)
        return ev

    def guests(self, k: int, exclude: str) -> list[tuple[str, str | None]]:
        pool = [self.email(p) for p in self.people if self.email(p) != exclude]
        return [(e, self.rng.choice(PARTSTATS)) for e in self.rng.sample(pool, min(k, len(pool)))]

    def occurrences(self, ev: dict, horizon: dt.datetime) -> list[dt.datetime]:
        return list(_starts(ev["start"], ev["rule"], lambda w: _utc(w, ev["zone"]), set(), horizon))

    def serialise(self, stamp: str, zones: list[str]) -> str:
        rng = self.rng
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Northwind//Groupware Calendar Export 4.2//EN",
                 "CALSCALE:GREGORIAN", "METHOD:PUBLISH", "X-WR-CALNAME:Team calendar"]
        for zone in zones:
            lines += _vtimezone(zone)
        order = list(self.events)
        rng.shuffle(order)
        for ev in order:
            body = ["BEGIN:VEVENT", f"UID:{ev['uid']}", f"DTSTAMP:{stamp}"]
            if ev["allday_days"]:
                day = ev["start"]
                body += [f"DTSTART;VALUE=DATE:{day:%Y%m%d}",
                         f"DTEND;VALUE=DATE:{day + dt.timedelta(days=ev['allday_days']):%Y%m%d}"]
            else:
                body.append(self.dtprop("DTSTART", ev["start"], ev["zone"]))
                if ev["duration_prop"]:
                    h, m = divmod(ev["minutes"], 60)
                    body.append("DURATION:PT" + (f"{h}H" if h else "") + (f"{m}M" if m else ""))
                else:
                    body.append(self.dtprop("DTEND", ev["start"] + dt.timedelta(minutes=ev["minutes"]), ev["zone"]))
            if ev["rule"]:
                body.append("RRULE:" + self.rrule(ev["rule"]))
            if ev["exdates"]:
                if ev["exdate_utc"]:
                    body.append("EXDATE:" + ",".join(f"{_utc(x, ev['zone']):%Y%m%dT%H%M%SZ}" for x in ev["exdates"]))
                else:
                    body.append(f"EXDATE;TZID={ev['zone']}:" + ",".join(f"{x:%Y%m%dT%H%M%S}" for x in ev["exdates"]))
            body.append(f"SUMMARY:{ev['summary']}")
            if rng.random() < 0.6:
                body.append(f"LOCATION:{rng.choice(LOCATIONS)}")
            if rng.random() < 0.3:
                body.append("DESCRIPTION:Agenda: updates\\, risks\\, next steps. Dial-in details are in the invite"
                            "\\; please join five minutes early.")
            if ev["status"]:
                body.append(f"STATUS:{ev['status']}")
            body.append(f'ORGANIZER;CN="{self.cn(ev["organizer"])}":mailto:{ev["organizer"]}')
            for email, partstat in ev["attendees"]:
                params = [f'CN="{self.cn(email)}"', "CUTYPE=INDIVIDUAL", "ROLE=REQ-PARTICIPANT"]
                if partstat:
                    params.insert(1 + rng.randrange(2), f"PARTSTAT={partstat}")
                prefix = "MAILTO:" if rng.random() < 0.2 else "mailto:"
                body.append("ATTENDEE;" + ";".join(params) + ";RSVP=TRUE:" + prefix + email)
            body += ["SEQUENCE:" + str(rng.randint(0, 3)), "END:VEVENT"]
            lines += body
        lines.append("END:VCALENDAR")
        return "\r\n".join(part for line in lines for part in _fold(line)) + "\r\n"

    def cn(self, email: str) -> str:
        first, last = email.split("@")[0].split(".")
        return f"{last.capitalize()}, {first.capitalize()}"

    @staticmethod
    def dtprop(name: str, wall: dt.datetime, zone: str) -> str:
        if zone == "UTC":
            return f"{name}:{wall:%Y%m%dT%H%M%S}Z"
        return f"{name};TZID={zone}:{wall:%Y%m%dT%H%M%S}"

    def rrule(self, rule: dict) -> str:
        parts = []
        if rule["INTERVAL"] > 1:
            parts.append(f"INTERVAL={rule['INTERVAL']}")
        if rule["BYDAY"]:
            parts.append("BYDAY=" + ",".join(DAYS[d] for d in rule["BYDAY"]))
        if rule["COUNT"] is not None:
            parts.append(f"COUNT={rule['COUNT']}")
        if rule["UNTIL"] is not None:
            parts.append(f"UNTIL={rule['UNTIL']:%Y%m%dT%H%M%SZ}")
        self.rng.shuffle(parts)
        return ";".join([f"FREQ={rule['FREQ']}", *parts])


def _rule_spec(freq: str, *, interval: int = 1, byday=(), count=None, until=None) -> dict:
    return {"FREQ": freq, "INTERVAL": interval, "BYDAY": list(byday), "COUNT": count, "UNTIL": until}


def _calendar_build(level: int, rng: random.Random) -> Built:
    year = rng.choice(YEARS)
    us, eu = rng.choice(US_ZONES), rng.choice(EU_ZONES)
    if rng.random() < 0.5:  # spring: the US moves first; autumn: the EU moves back first
        gap_from, gap_to = _change(us, year, 3), _change(eu, year, 3)
    else:
        gap_from, gap_to = _change(eu, year, 10), _change(us, year, 11)
    first = gap_from - dt.timedelta(days=gap_from.weekday() + 7 * rng.randint(1, 2))
    last = gap_to - dt.timedelta(days=gap_to.weekday()) + dt.timedelta(weeks=rng.randint(0, 1))
    report = rng.choice(REPORT_ZONES)
    company = rng.choice(COMPANIES)
    staff = [(f, l, company) for f, l in rng.sample(NAMES, rng.randint(8, 11))]
    asked = rng.sample(staff, rng.randint(3, 5))
    g = _Gen(rng, staff, first - dt.timedelta(days=7), last + dt.timedelta(days=10))
    in_lo, in_hi = first, last + dt.timedelta(days=7)
    horizon = dt.datetime.combine(in_hi + dt.timedelta(days=3), dt.time(), UTC)

    def someone() -> tuple[str, str, str]:
        return rng.choice(asked)

    # The daylight-saving pair: a weekly US series and a weekly EU series at the same instant outside the gap.
    p = someone()
    anchor = dt.datetime.combine(first + dt.timedelta(days=rng.randrange(5)), dt.time(rng.randint(8, 11),
                                                                                       rng.choice((0, 15, 30))))
    eu_wall = _utc(anchor, us).astimezone(ZoneInfo(eu)).replace(tzinfo=None)
    day = anchor.weekday()
    for wall, zone in ((anchor, us), (eu_wall, eu)):
        organizer = g.email(p) if zone == us else None
        att = g.guests(rng.randint(1, 3), g.email(p))
        if organizer is None:
            att.append((g.email(p), "ACCEPTED"))
        g.add(start=wall, zone=zone, minutes=rng.choice((30, 45, 60)), rule=_rule_spec("WEEKLY", byday=[day]),
              organizer=organizer, attendees=att)

    # A series ending on UNTIL exactly at one of its occurrences, inside the range.
    p = someone()
    zone = rng.choice(EVENT_ZONES)
    ev = g.add(start=g.wall(in_lo, in_lo + dt.timedelta(days=5)), zone=zone, minutes=rng.choice(MINUTES),
               rule=_rule_spec("WEEKLY"), organizer=g.email(p), attendees=g.guests(rng.randint(1, 3), g.email(p)))
    occ = g.occurrences(ev, horizon)
    ev["rule"]["UNTIL"] = _utc(occ[rng.randint(1, 2)], zone)

    # A daily series with COUNT whose second occurrence is excluded (COUNT still counts it).
    p = someone()
    zone = rng.choice(EVENT_ZONES)
    ev = g.add(start=g.wall(in_lo, in_lo + dt.timedelta(days=10)), zone=zone, minutes=rng.choice(MINUTES),
               rule=_rule_spec("DAILY", count=rng.randint(3, 6)), organizer=g.email(p),
               attendees=g.guests(rng.randint(1, 2), g.email(p)))
    ev["exdates"] = [g.occurrences(ev, horizon)[1]]

    # A weekly series with an excluded week inside the range.
    p = someone()
    zone = rng.choice(EVENT_ZONES)
    ev = g.add(start=g.wall(in_lo - dt.timedelta(days=7), in_lo + dt.timedelta(days=3)), zone=zone,
               minutes=rng.choice(MINUTES), rule=_rule_spec("WEEKLY", count=rng.randint(4, 8)),
               attendees=[(g.email(p), "ACCEPTED")] + g.guests(rng.randint(0, 2), g.email(p)))
    ev["exdates"] = [w for w in g.occurrences(ev, horizon) if in_lo <= w.date() < in_hi][1:2]

    # A cancelled event, an all-day event, a declined invitation and an attendee written in other letter case.
    p = someone()
    g.add(start=g.wall(in_lo, in_hi), zone=rng.choice(EVENT_ZONES), minutes=rng.choice(MINUTES), status="CANCELLED",
          organizer=g.email(p), attendees=g.guests(rng.randint(1, 3), g.email(p)))
    p = someone()
    g.add(start=g.wall(in_lo, in_hi).date(), allday_days=rng.randint(1, 2), organizer=g.email(p),
          attendees=g.guests(rng.randint(1, 4), g.email(p)))
    p = someone()
    g.add(start=g.wall(in_lo, in_hi), zone=rng.choice(EVENT_ZONES), minutes=rng.choice(MINUTES),
          organizer=g.email(rng.choice([s for s in staff if s != p])),
          attendees=[(g.email(p), "DECLINED")] + g.guests(rng.randint(1, 2), g.email(p)))
    p = someone()
    g.add(start=g.wall(in_lo, in_hi), zone=rng.choice(EVENT_ZONES), minutes=rng.choice(MINUTES),
          organizer=g.email(rng.choice([s for s in staff if s != p])),
          attendees=[(g.email(p, shout=True), "ACCEPTED")] + g.guests(rng.randint(1, 2), g.email(p)))

    # Background: single events, weekly and daily series, all-day events.
    for _ in range(rng.randint(55, 80)):
        zone = rng.choice(EVENT_ZONES)
        organizer = g.email(rng.choice(staff))
        att = g.guests(rng.randint(1, 4), organizer)
        roll = rng.random()
        if roll < 0.08:
            g.add(start=g.wall().date(), allday_days=rng.randint(1, 3), organizer=organizer, attendees=att,
                  status=rng.choice(STATUSES))
            continue
        ev = g.add(start=g.wall(hours=(7, 18)), zone=zone, minutes=rng.choice(MINUTES), organizer=organizer,
                   attendees=att, status=rng.choice(STATUSES))
        if roll < 0.5:
            continue
        if roll < 0.85:
            days = sorted({ev["start"].weekday(), *rng.sample(range(5), rng.randint(0, 2))})
            ev["rule"] = _rule_spec("WEEKLY", interval=rng.choice((1, 1, 1, 2)), byday=days)
        else:
            ev["rule"] = _rule_spec("DAILY", interval=rng.choice((1, 1, 2)))
        occ = g.occurrences(ev, horizon)
        ending = rng.random()
        if ending < 0.4 and len(occ) > 2:
            ev["rule"]["COUNT"] = rng.randint(2, min(len(occ), 12))
        elif ending < 0.75 and len(occ) > 2:
            k = rng.randint(1, min(len(occ), 12) - 1)
            ev["rule"]["UNTIL"] = _utc(occ[k], zone) + dt.timedelta(hours=rng.choice((0, 0, 0, 2, 30)))
        if rng.random() < 0.35 and len(occ) > 3:
            ev["exdates"] = sorted(rng.sample(occ[: min(len(occ), 10)], rng.randint(1, 2)))

    zones = sorted({ev["zone"] for ev in g.events if not ev["allday_days"] and ev["zone"] != "UTC"})
    stamp = f"{in_lo - dt.timedelta(days=rng.randint(1, 20)):%Y%m%d}T{rng.randint(0, 23):02d}0000Z"
    request = {"people": [g.email(p) for p in asked], "week_from": f"{first.isocalendar()[0]}-W{first.isocalendar()[1]:02d}",
               "week_to": f"{last.isocalendar()[0]}-W{last.isocalendar()[1]:02d}", "time_zone": report}
    files = {FILE: g.serialise(stamp, zones), "request.json": json.dumps(request, indent=2) + "\n"}
    statement = "\n".join([
        f"The operations team exported the shared calendar as {FILE} (iCalendar, RFC 5545). request.json lists "
        "people (email addresses), a range of ISO weeks (week_from to week_to, both included) and a report "
        "time_zone. For each person and week, report meeting hours and double bookings.",
        "",
        "Rules:",
        "- Unfold folded lines first: a line that starts with a space continues the previous line. Only VEVENT "
        "blocks are events; VTIMEZONE blocks describe the zones, which are IANA names.",
        "- Expand RRULE (FREQ=DAILY or WEEKLY, with INTERVAL, BYDAY, COUNT or UNTIL). COUNT counts occurrences "
        "before EXDATE removes any; UNTIL is in UTC and inclusive. Remove every occurrence listed in EXDATE, "
        "whether written in local time with a TZID or in UTC.",
        "- A time has a TZID or ends in Z (UTC). An occurrence keeps its wall-clock time in its zone across "
        "daylight-saving changes. Compare times as real instants.",
        "- Skip events with STATUS:CANCELLED. All-day events (VALUE=DATE) are not meetings: no hours, no conflicts.",
        "- A person attends when they are the ORGANIZER or an ATTENDEE whose PARTSTAT is not DECLINED. Email "
        "addresses match case-insensitively.",
        "- An occurrence lasts DTEND minus DTSTART, or its DURATION, and belongs to the ISO week of its start in "
        "the report time_zone.",
        "- meeting_hours: the person's summed occurrence lengths in that week, in hours, rounded to 2 decimals.",
        "- conflicts: pairs of the person's occurrences that overlap (one starts before the other ends; touching "
        "is not a conflict), counted in the week of the earlier start.",
        "",
        "Write meeting_hours.csv with exactly this header line: " + ",".join(HEADER) + ", then one row per person "
        "in request order and, within a person, one per week in order (weeks written like 2026-W09), zeros "
        "included. The file is too long to read by eye: use a script (zoneinfo is available).",
    ])
    built = Built(files=files, statement=statement, output=OUTPUT, reference=None)
    built.reference = rows(_calendar_solve(files, level, None))
    return built


FAMILIES = (
    Family("calendar", DOMAIN, _calendar_build, _calendar_check, _calendar_solve,
           {trap: (LEVEL,) for trap in ("ignore_exdate", "ignore_tz", "fixed_offset", "keep_cancelled",
                                        "allday_24h", "until_exclusive", "count_declined", "no_unfold",
                                        "case_sensitive", "count_skips_exdate")},
           levels=(LEVEL,)),
)
