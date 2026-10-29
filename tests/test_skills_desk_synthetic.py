"""Synthetic desk families (level 4): the calendar (office) and authlog (security) readers, rules and contracts."""

from __future__ import annotations

import datetime as dt
import json
import random

from qe_arena import skills
from qe_arena.skills import desk_authlog, desk_office

UTC = dt.timezone.utc


def _task(cell: str, kind: str, seed: int = 0) -> skills.SkillTask:
    return skills.make_skill_task(cell, seed, pool=(kind,))


def _ics(*events: list[str]) -> str:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "BEGIN:VTIMEZONE", "TZID:Europe/Berlin", "BEGIN:STANDARD",
             "DTSTART:19701025T030000", "END:STANDARD", "END:VTIMEZONE"]
    for body in events:
        lines += ["BEGIN:VEVENT", *body, "END:VEVENT"]
    return "\r\n".join([*lines, "END:VCALENDAR"]) + "\r\n"


def _calendar(events: list[list[str]], people=("ana@x.example",), week_from="2026-W10", week_to="2026-W11",
              zone="Europe/Berlin", variant=None) -> list[list[str]]:
    files = {desk_office.FILE: _ics(*events), "request.json": json.dumps(
        {"people": list(people), "week_from": week_from, "week_to": week_to, "time_zone": zone})}
    return [line.split(",") for line in desk_office._calendar_solve(files, 4, variant).splitlines()[1:]]


ANA = "ORGANIZER:mailto:ana@x.example"


# ---------------------------------------------------------------- calendar: reading

def test_calendar_unfolds_lines_and_splits_on_the_first_colon_outside_quotes():
    text = 'ATTENDEE;CN="Ruiz: Ana, PhD";PARTSTAT=DECLINED;ROLE=REQ-PART\r\n ICIPANT:MAILTO:Ana@X.example\r\n'
    (line,) = [x for x in desk_office._unfold(text) if x]
    name, params, value = desk_office._prop(line)
    assert name == "ATTENDEE" and params["CN"] == "Ruiz: Ana, PhD" and params["ROLE"] == "REQ-PARTICIPANT"
    assert params["PARTSTAT"] == "DECLINED" and value == "MAILTO:Ana@X.example"
    assert [x for x in desk_office._unfold(text, "no_unfold") if x] == [text.split("\r\n")[0]]


def test_calendar_reads_utc_local_and_date_values_and_durations():
    assert desk_office._when("20260310T150000Z", {}) == (dt.datetime(2026, 3, 10, 15), "UTC")
    assert desk_office._when("20260310T100000", {"TZID": "America/New_York"}) == \
        (dt.datetime(2026, 3, 10, 10), "America/New_York")
    assert desk_office._when("20260310", {"VALUE": "DATE"}) == dt.date(2026, 3, 10)
    assert desk_office._duration("PT1H30M") == dt.timedelta(minutes=90)
    assert desk_office._duration("PT50M") == dt.timedelta(minutes=50)
    assert desk_office._duration("P1D") == dt.timedelta(days=1)


def test_calendar_count_includes_exdates_and_until_is_inclusive():
    to_utc = lambda w: w.replace(tzinfo=UTC)  # noqa: E731
    start, far = dt.datetime(2026, 3, 2, 9), dt.datetime(2027, 1, 1, tzinfo=UTC)
    excluded = {dt.datetime(2026, 3, 3, 9, tzinfo=UTC)}
    rule = desk_office._rule("FREQ=DAILY;COUNT=4")
    assert [w.day for w in desk_office._starts(start, rule, to_utc, excluded, far)] == [2, 4, 5]
    assert [w.day for w in desk_office._starts(start, rule, to_utc, excluded, far, "count_skips_exdate")] == \
        [2, 4, 5, 6]
    rule = desk_office._rule("FREQ=WEEKLY;UNTIL=20260316T090000Z;BYDAY=MO")
    assert rule["UNTIL"] == (dt.datetime(2026, 3, 16, 9), "UTC")  # the solver turns it into an instant
    rule["UNTIL"] = rule["UNTIL"][0].replace(tzinfo=UTC)
    assert [w.day for w in desk_office._starts(start, rule, to_utc, set(), far)] == [2, 9, 16]
    assert [w.day for w in desk_office._starts(start, rule, to_utc, set(), far, "until_exclusive")] == [2, 9]
    rule = desk_office._rule("FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,TH;COUNT=5")
    assert [w.day for w in desk_office._starts(start, rule, to_utc, set(), far)] == [2, 5, 16, 19, 30]


def test_calendar_keeps_wall_clock_time_across_daylight_saving():
    # New York moves to EDT on 2026-03-08: a 10:00 weekly meeting is 15:00Z before and 14:00Z after.
    ev = ["DTSTART;TZID=America/New_York:20260302T100000", "DTEND;TZID=America/New_York:20260302T110000",
          "RRULE:FREQ=WEEKLY;COUNT=3", ANA]
    other = ["DTSTART:20260302T143000Z", "DURATION:PT1H", "RRULE:FREQ=WEEKLY;COUNT=3", ANA]
    got = _calendar([ev, other], week_from="2026-W10", week_to="2026-W12", zone="UTC")
    assert [r[3] for r in got] == ["1", "1", "1"]  # overlap 15:00-15:30Z, then 14:30-15:00Z
    london = ["DTSTART;TZID=Europe/London:20260302T150000", "DURATION:PT1H", "RRULE:FREQ=WEEKLY;COUNT=3", ANA]
    got = _calendar([ev, london], week_from="2026-W10", week_to="2026-W12", zone="UTC")
    assert [r[3] for r in got] == ["1", "0", "0"]  # London is still on GMT until 2026-03-29
    assert [r[3] for r in _calendar([ev, london], week_from="2026-W10", week_to="2026-W12", zone="UTC",
                                    variant="fixed_offset")] == ["1", "1", "1"]


def test_calendar_skips_cancelled_all_day_and_declined_and_matches_email_case_insensitively():
    base = ["DTSTART;TZID=Europe/Berlin:20260303T100000", "DTEND;TZID=Europe/Berlin:20260303T104500"]
    events = [
        [*base, "ORGANIZER:mailto:bob@x.example", "ATTENDEE;PARTSTAT=ACCEPTED:MAILTO:Ana@X.Example"],
        [*base, "STATUS:CANCELLED", ANA],
        ["DTSTART;VALUE=DATE:20260304", "DTEND;VALUE=DATE:20260305", ANA],
        [*base, "ORGANIZER:mailto:bob@x.example", "ATTENDEE;PARTSTAT=DECLINED:mailto:ana@x.example"],
        [*base, "ORGANIZER:mailto:bob@x.example", "ATTENDEE:mailto:ana@x.example"],  # no PARTSTAT: attends
    ]
    assert _calendar(events)[0][2:] == ["1.50", "1"]
    assert _calendar(events, variant="keep_cancelled")[0][2:] == ["2.25", "3"]
    assert _calendar(events, variant="allday_24h")[0][2:] == ["25.50", "1"]
    assert _calendar(events, variant="count_declined")[0][2:] == ["2.25", "3"]
    assert _calendar(events, variant="case_sensitive")[0][2:] == ["0.75", "0"]


def test_calendar_touching_meetings_do_not_conflict_and_weeks_follow_the_report_zone():
    a = ["DTSTART:20260303T090000Z", "DTEND:20260303T100000Z", ANA]
    b = ["DTSTART:20260303T100000Z", "DTEND:20260303T103000Z", ANA]
    assert _calendar([a, b])[0][2:] == ["1.50", "0"]
    # Sunday 2026-03-08 23:30Z is Monday 00:30 in Berlin: week 11 there, week 10 in UTC.
    late = ["DTSTART:20260308T233000Z", "DURATION:PT20M", ANA]
    assert [r[2] for r in _calendar([late])] == ["0.00", "0.33"]
    assert [r[2] for r in _calendar([late], zone="UTC")] == ["0.33", "0.00"]


def test_calendar_exdate_matches_in_utc_or_local_form():
    series = ["DTSTART;TZID=Europe/Berlin:20260302T090000", "DURATION:PT1H", "RRULE:FREQ=DAILY;COUNT=3", ANA]
    assert _calendar([series])[0][2] == "3.00"
    assert _calendar([[*series, "EXDATE:20260303T080000Z"]])[0][2] == "2.00"
    assert _calendar([[*series, "EXDATE;TZID=Europe/Berlin:20260303T090000,20260304T090000"]])[0][2] == "1.00"
    assert _calendar([[*series, "EXDATE:20260303T080000Z"]], variant="ignore_exdate")[0][2] == "3.00"


# ---------------------------------------------------------------- calendar: the contract

def test_calendar_hours_accept_their_rounding_and_reject_more():
    task = _task("qe_skill-office-d4", "calendar", 2)
    header, *body = skills.solve(task).splitlines()
    cells = body[0].split(",")
    for delta, expected in ((0.004, 1.0), (-0.004, 1.0), (0.01, 0.0)):
        moved = ",".join([cells[0], cells[1], f"{float(cells[2]) + delta:.3f}", cells[3]])
        assert skills.grade(task, "\n".join([header, moved, *body[1:]]) + "\n") == expected, delta
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 0.0
    assert skills.grade(task, "\n".join([header.upper(), *body]) + "\n") == 0.0


def test_calendar_range_spans_the_daylight_saving_gap_and_every_person_week_is_a_row():
    for seed in range(20):
        task = _task("qe_skill-office-d4", "calendar", seed)
        request = json.loads(task.files["request.json"])
        weeks = sorted({r["week"] for r in task.private["reference"]})
        assert len(task.private["reference"]) == len(request["people"]) * len(weeks) and len(weeks) >= 3
        assert weeks[0] == request["week_from"] and weeks[-1] == request["week_to"]
        assert "BEGIN:VTIMEZONE" in task.files[desk_office.FILE]


# ---------------------------------------------------------------- authlog: reading

def test_authlog_years_roll_over_and_days_are_space_padded():
    text = ("Dec 31 23:59:58 h sshd[1]: x\nJan  1 00:00:03 h sshd[1]: y\nJan 10 00:00:03 h sshd[1]: z\n")
    assert [w for w, _ in desk_authlog._lines(text, 2024, None)] == [
        dt.datetime(2024, 12, 31, 23, 59, 58), dt.datetime(2025, 1, 1, 0, 0, 3), dt.datetime(2025, 1, 10, 0, 0, 3)]
    assert [w.year for w, _ in desk_authlog._lines(text, 2024, "year_fixed")] == [2024, 2024, 2024]


def test_authlog_reads_each_line_kind():
    f = desk_authlog._failure
    assert f("sshd[9]: Failed password for root from 203.0.113.5 port 22 ssh2", None) == "203.0.113.5"
    invalid = "sshd[9]: Failed password for invalid user pi from 2001:db8::1f port 22 ssh2"
    assert f(invalid, None) == "2001:db8::1f" and f(invalid, "skip_invalid_user") is None
    line = "sshd[9]: Invalid user pi from 2001:db8::1f port 22"
    assert f(line, None) is None and f(line, "count_invalid_lines") == "2001:db8::1f"
    pam = "sshd[9]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=1.2.3.4"
    assert f(pam, None) is None and f(pam, "count_pam") == "1.2.3.4"
    sudo = "sudo: pam_unix(sudo:auth): authentication failure; logname=a uid=1 euid=0 tty=/dev/pts/0 ruser=a rhost=  user=a"
    assert f(sudo, "count_pam") is None


def test_authlog_sliding_window_is_inclusive_and_not_clock_aligned():
    t0 = dt.datetime(2024, 12, 31, 10, 4, 0)
    times = [t0 + dt.timedelta(seconds=s) for s in (0, 50, 100, 300)]  # 4 failures, exactly 5 minutes wide
    assert desk_authlog._first_burst(times, 4, 5, None) == t0
    assert desk_authlog._first_burst(times, 4, 5, "window_exclusive") is None
    straddle = [t0 + dt.timedelta(seconds=s) for s in (40, 50, 70, 80)]  # 10:04:40 .. 10:05:20
    assert desk_authlog._first_burst(straddle, 4, 5, None) == straddle[0]
    assert desk_authlog._first_burst(straddle, 4, 5, "fixed_window") is None


def _log(lines: str, threshold=3, window=5, year=2024, variant=None) -> list[list[str]]:
    files = {desk_authlog.FILE: lines, "request.json": json.dumps(
        {"year": year, "threshold": threshold, "window_minutes": window})}
    return [r.split(",") for r in desk_authlog._authlog_solve(files, 4, variant).splitlines()[1:]]


def test_authlog_repeated_messages_expand_and_success_must_follow_the_burst():
    lines = ("Dec 31 23:58:00 h sshd[1]: Accepted publickey for deploy from 198.51.100.9 port 1 ssh2\n"
             "Dec 31 23:59:00 h sshd[2]: Failed password for root from 198.51.100.9 port 2 ssh2\n"
             "Dec 31 23:59:01 h sshd[2]: message repeated 2 times: [ Failed password for root from 198.51.100.9 port 2 ssh2]\n"
             "Jan  1 00:01:00 h sshd[3]: Failed password for invalid user pi from 2001:db8::7 port 3 ssh2\n"
             "Jan  1 00:01:30 h sshd[3]: Failed password for invalid user pi from 2001:db8::7 port 3 ssh2\n"
             "Jan  1 00:02:00 h sshd[3]: Failed password for invalid user pi from 2001:db8::7 port 3 ssh2\n"
             "Jan  1 00:09:00 h sshd[4]: Accepted password for pi from 2001:db8::7 port 4 ssh2\n")
    assert _log(lines) == [["198.51.100.9", "2024-12-31T23:59:00", "3", "no"],
                           ["2001:db8::7", "2025-01-01T00:01:00", "3", "yes"]]
    assert _log(lines, variant="repeat_once") == [["2001:db8::7", "2025-01-01T00:01:00", "3", "yes"]]
    assert _log(lines, variant="success_anytime")[0][3] == "yes"
    assert _log(lines, variant="ipv4_only") == [["198.51.100.9", "2024-12-31T23:59:00", "3", "no"]]


def test_authlog_failures_are_exact_strings_and_rows_keep_their_order():
    task = _task("qe_skill-security-d4", "authlog", 1)
    header, *body = skills.solve(task).splitlines()
    cells = body[0].split(",")
    assert skills.grade(task, "\n".join([header, ",".join([*cells[:2], cells[2] + ".0", cells[3]]), *body[1:]])) == 0
    assert skills.grade(task, "\n".join([header, ",".join([*cells[:3], cells[3].upper()]), *body[1:]])) == 0
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 0.0
    assert skills.grade(task, "\n".join([header, *body]) + "\n") == 1.0


def test_authlog_crosses_new_year_and_has_ipv6_and_repeats():
    for seed in range(20):
        task = _task("qe_skill-security-d4", "authlog", seed)
        log = task.files[desk_authlog.FILE]
        assert log.startswith("Dec ") and "\nJan  1 " in log and "message repeated" in log
        ref = task.private["reference"]
        assert any(":" in r["ip"] for r in ref) and any(r["burst_start"].endswith(("-01-01", "-01-02"), 0, 10)
                                                         for r in ref)
        assert {r["succeeded"] for r in ref} == {"yes", "no"}


# ---------------------------------------------------------------- both: the rules bind on the data, not by retry

def test_every_listed_trap_binds_on_almost_every_raw_build():
    for module in (desk_office, desk_authlog):
        fam = module.FAMILIES[0]
        clean = 0
        for seed in range(60):
            built = fam.build(4, random.Random(f"bind:{seed}"))
            assert all(fam.check(built.reference, fam.solve(built.files, 4, None)))
            clean += all(not all(fam.check(built.reference, fam.solve(built.files, 4, trap))) for trap in fam.traps)
        assert clean >= 57, (fam.name, clean)  # >= 95%: no seed comes near MAX_ATTEMPTS
