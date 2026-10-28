"""v9 desk families on pinned real data: USGS earthquake declustering (quakes) and NOAA GML annual means (ghg)."""

from __future__ import annotations

import json

from qe_arena import skills
from qe_arena.skills import desk_ghg, desk_industrial

QUAKES, GHG = "qe_skill-industrial-d4", "qe_skill-science-d4"
CSV_HEAD = ("time,latitude,longitude,depth,mag,magType,nst,gap,dmin,rms,net,id,updated,place,type,horizontalError,"
            "depthError,magError,magNst,status,locationSource,magSource")


def _task(cell: str, kind: str, seed: int = 0) -> skills.SkillTask:
    return skills.make_skill_task(cell, seed, pool=(kind,))


def _event(time: str, lat: float, lon: float, mag: str, eid: str, *, mag_type: str = "md", kind: str = "earthquake",
           status: str = "reviewed") -> str:
    return f'{time},{lat},{lon},5.0,{mag},{mag_type},,,,,nc,{eid},{time},"1km N of Somewhere, CA",{kind},,,,,' \
           f"{status},nc,nc"


def _quake_files(events: list[str], **request) -> dict[str, str]:
    base = {"start": "2026-01-01T00:00:00.000Z", "end": "2026-02-01T00:00:00.000Z", "min_magnitude": 1.5,
            "mag_types": ["md", "ml"],
            "windows": [{"min_mag": 0.0, "distance_km": 10.0, "days": 2.0},
                        {"min_mag": 3.0, "distance_km": 20.0, "days": 10.0}]}
    return {desk_industrial.FILE: "\n".join([CSV_HEAD, *events]) + "\n",
            "request.json": json.dumps({**base, **request})}


def test_both_briefs_fit_the_desk_shape_and_name_the_exact_header():
    for cell, kind, module in ((QUAKES, "quakes", desk_industrial), (GHG, "ghg", desk_ghg)):
        for seed in range(20):
            task = _task(cell, kind, seed)
            assert 1200 <= len(task.prompt) <= 2000, (kind, seed, len(task.prompt))
            assert ",".join(module.HEADER) in task.prompt
            assert max(len(text) for text in task.files.values()) > 2 * 8192
            assert set(task.files) == {"request.json", next(n for n in task.files if n != "request.json")}


def test_quakes_serves_a_pinned_slice_verbatim_newest_first():
    task = _task(QUAKES, "quakes", 1)
    text = task.files[desk_industrial.FILE]
    assert text in {desk_industrial._snapshot(name) for name in desk_industrial._slices()}
    events = desk_industrial._events(text)
    assert events[0].t > events[-1].t
    assert {e.status for e in events} == {"reviewed", "automatic"} and len({e.kind for e in events}) > 1


def test_quakes_requests_are_chosen_so_the_rules_bind():
    for seed in range(20):
        task = _task(QUAKES, "quakes", seed)
        request = json.loads(task.files["request.json"])
        events = desk_industrial._events(task.files[desk_industrial.FILE])
        times = {e.time for e in events}
        assert request["start"] in times and request["end"] in times
        start, end = desk_industrial._seconds(request["start"]), desk_industrial._seconds(request["end"])
        assert any(e.mag == request["min_magnitude"] and start <= e.t < end for e in events)
        assert {e.mag_type for e in events} - set(request["mag_types"])  # some magnitude type is left out


def test_haversine_and_the_window_table():
    a = desk_industrial.Event("", 0, 38.0, -122.0, 0, 2, "2", "md", "a", "earthquake", "reviewed")
    north = a._replace(lat=39.0)
    east = a._replace(lon=-121.0)
    assert abs(desk_industrial._haversine(a, north) - 111.195) < 0.01
    assert abs(desk_industrial._haversine(a, east) - 87.6) < 0.1  # one degree of longitude at 38 N
    assert desk_industrial._flat(a, east) > 111
    table = [[0.0, 10.0, 2.0], [2.5, 19.5, 6.0], [3.0, 22.5, 11.5]]
    assert desk_industrial._window(table, 2.49) == (10.0, 2.0)
    assert desk_industrial._window(table, 2.5) == (19.5, 6.0)
    assert desk_industrial._window(table, 7.0) == (22.5, 11.5)


def test_quakes_declustering_rules_on_a_small_catalogue():
    events = [  # newest first, as the service returns them
        _event("2026-01-20T00:00:00.000Z", 38.0, -122.0, "1.9", "late"),  # 19 days after main: outside its window
        _event("2026-01-05T00:00:00.000Z", 38.05, -122.0, "2.0", "after"),  # 5.6 km, 4 days: aftershock of main
        _event("2026-01-04T00:00:00.000Z", 38.0, -122.0, "3.2", "bigger", mag_type="mw"),  # type not used
        _event("2026-01-03T00:00:00.000Z", 38.0, -122.0, "2.5", "auto", status="automatic"),
        _event("2026-01-02T12:00:00.000Z", 38.0, -122.01, "1.8", "blast", kind="quarry blast"),
        _event("2026-01-02T00:00:00.000Z", 38.3, -122.0, "1.7", "far"),  # 33 km away: its own mainshock
        _event("2026-01-01T00:00:00.000Z", 38.0, -122.0, "3.0", "main"),  # on the start time: included
        _event("2026-02-01T00:00:00.000Z", 38.0, -122.0, "2.2", "at_end"),  # on the end time: excluded
    ]
    out = desk_industrial._quakes_solve(_quake_files(events), 4, None).splitlines()
    assert out == ["id,time,mag,magType,aftershocks", "main,2026-01-01T00:00:00.000Z,3.0,md,1",
                   "far,2026-01-02T00:00:00.000Z,1.7,md,0", "late,2026-01-20T00:00:00.000Z,1.9,md,0"]


def test_quakes_counts_an_aftershock_for_the_largest_qualifying_mainshock_and_never_for_a_smaller_one():
    events = [
        _event("2026-01-03T00:00:00.000Z", 38.0, -122.0, "2.1", "c"),  # within both mainshocks' windows
        _event("2026-01-02T00:00:00.000Z", 38.0, -122.15, "2.6", "b"),  # 13 km from a: a mainshock of its own
        _event("2026-01-01T12:00:00.000Z", 38.0, -122.0, "2.8", "bigger_than_none"),
        _event("2026-01-01T00:00:00.000Z", 38.0, -122.0, "2.4", "a"),
    ]
    out = desk_industrial._quakes_solve(_quake_files(events), 4, None).splitlines()[1:]
    # bigger_than_none is larger than a, so it cannot be a's aftershock; c goes to the largest qualifying mainshock
    assert [line.split(",")[0] + ":" + line.split(",")[-1] for line in out] == [
        "a:0", "bigger_than_none:1", "b:0"]


def test_quakes_rows_keep_time_order_and_magnitudes_compare_as_numbers():
    task = next(t for t in (_task(QUAKES, "quakes", s) for s in range(10)) if len(t.private["reference"]) > 2)
    header, *body = skills.solve(task).splitlines()
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 0.0
    cells = body[0].split(",")
    padded = ",".join([*cells[:2], f"{float(cells[2]):.3f}", *cells[3:]])
    assert skills.grade(task, "\n".join([header, padded, *body[1:]]) + "\n") == 1.0
    moved = ",".join([*cells[:2], f"{float(cells[2]) + 0.01:.2f}", *cells[3:]])
    assert skills.grade(task, "\n".join([header, moved, *body[1:]]) + "\n") == 0.0


GML_HEAD = "# header_lines : 3\n# value:_FillValue : -999.999\nsite_code year month value nvalue qcflag\n"


def test_ghg_reads_fill_values_flags_and_day_counts():
    text = GML_HEAD + "BRW 2001 1 400.10 20 ...\nBRW 2001 2 -999.99 0 *..\nBRW 2001 3 401.00 5 ...\n" \
                      "BRW 2001 4 402.00 25 *..\n"
    months = desk_ghg._months(text)
    assert [(m.value, m.days, m.flag) for m in months] == [
        (400.1, 20, "..."), (-999.99, 0, "*.."), (401.0, 5, "..."), (402.0, 25, "*..")]
    assert [desk_ghg._qualifies(m, 10) for m in months] == [True, False, False, False]


def test_ghg_growth_needs_both_years_complete_and_reaches_before_the_window():
    lines = [f"SMO {year} {month} {300 + year - 2000 + month / 100:.2f} 25 ..."
             for year in (2000, 2001, 2003) for month in range(1, 13)]
    lines += [f"SMO 2002 {month} 310.00 {25 if month < 4 else 3} ..." for month in range(1, 13)]
    files = {"co2_smo_surface-insitu_1_ccgg_MonthlyData.txt": GML_HEAD + "\n".join(lines) + "\n",
             "request.json": json.dumps({"from_year": 2001, "to_year": 2003, "min_days": 10, "min_months": 10})}
    assert desk_ghg._ghg_solve(files, 4, None).splitlines()[1:] == [
        "2001,12,301.06,0.11,1.00", "2002,3,,,", "2003,12,303.06,0.11,"]  # 301.065 is half-way


def test_ghg_accepts_any_half_way_rounding_and_rejects_more():
    task = _task(GHG, "ghg", 2)
    text = skills.solve(task)
    ref = next(r for r in task.private["reference"] if r["mean"])
    rounded = next(line for line in text.splitlines() if line.startswith(ref["year"] + ","))
    mean = float(ref["mean"])
    for delta, expected in ((0.004, 1.0), (-0.004, 1.0), (0.007, 0.0)):
        cells = rounded.split(",")
        cells[2] = f"{mean + delta:.4f}"
        assert skills.grade(task, text.replace(rounded, ",".join(cells), 1)) == expected, delta


def test_ghg_incomplete_years_stay_empty_and_years_ascend():
    task = _task(GHG, "ghg", 5)
    text = skills.solve(task)
    header, *body = text.splitlines()
    gap = next(line for line in body if line.endswith(",,,"))
    assert skills.grade(task, text.replace(gap, gap[:-3] + ",1.00,1.00,", 1)) == 0.0
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 0.0
    assert [int(line.split(",")[0]) for line in body] == sorted(int(line.split(",")[0]) for line in body)
    request = json.loads(task.files["request.json"])
    assert body[0].split(",")[0] == str(request["from_year"]) and body[0].split(",")[-1] != ""
