"""Desk-tier (level 4) checks: real pinned data, stated rounding, and rules that are not required traps."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from qe_arena import skills
from fractions import Fraction

from qe_arena.skills import desk_finance, desk_math, desk_science, desk_security

DATA = Path(skills.__file__).resolve().parent / "data"


def _task(cell: str, kind: str, seed: int = 0) -> skills.SkillTask:
    return skills.make_skill_task(cell, seed, pool=(kind,))


def _swap(text: str, old: str, new: str) -> str:
    assert old in text
    return text.replace(old, new, 1)


def test_pinned_data_matches_its_provenance_and_names_a_licence():
    records = json.loads((DATA / "PROVENANCE.json").read_text())
    assert {r["path"] for r in records} == {str(p.relative_to(DATA)) for p in DATA.rglob("*.gz")}
    for record in records:
        raw = gzip.decompress((DATA / record["path"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == record["sha256"], record["path"]
        assert record["licence"] and record["url"].startswith("https://")


def test_desk_cells_are_not_served_before_calibration():
    for cell in ("qe_skill-science-d4", "qe_skill-math-d4", "qe_skill-security-d4", "qe_skill-finance-d4"):
        assert cell in skills.ALL_SKILL_IDS and cell not in skills.SKILL_TASK_IDS


def test_desk_briefs_are_long_and_the_inputs_too_big_to_print():
    for cell, kind in (("qe_skill-science-d4", "streamflow"), ("qe_skill-security-d4", "kev"),
                       ("qe_skill-finance-d4", "yields"), ("qe_skill-math-d4", "ledger")):
        for seed in range(10):
            task = _task(cell, kind, seed)
            assert len(task.prompt) > 1200, (kind, seed)
            assert max(len(text) for text in task.files.values()) > 2 * 8192, (kind, seed)  # 2x the output cap


def test_streamflow_accepts_its_rounding_and_rejects_more():
    task = _task("qe_skill-science-d4", "streamflow")
    text = skills.solve(task)
    row = next(r for r in task.private["reference"] if r["mean_flow"])
    mean = float(row["mean_flow"])
    for delta, expected in ((0.0005, 1.0), (-0.0005, 1.0), (0.002, 0.0)):
        assert skills.grade(task, _swap(text, f",{row['mean_flow']},", f",{mean + delta:.4f},")) == expected, delta


def test_streamflow_incomplete_years_must_stay_empty():
    task = _task("qe_skill-science-d4", "streamflow")
    text = skills.solve(task)
    row = next(r for r in task.private["reference"] if not r["mean_flow"])
    line = f"{row['water_year']},{row['approved_days']},,,"
    assert skills.grade(task, _swap(text, line, line[:-3] + ",0,0,")) == 0.0


def test_streamflow_rows_may_come_in_any_order_but_need_the_exact_header():
    task = _task("qe_skill-science-d4", "streamflow", 3)
    header, *body = skills.solve(task).splitlines()
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 1.0
    assert skills.grade(task, "\n".join([header.upper(), *body]) + "\n") == 0.0


def test_streamflow_treats_ice_and_provisional_values_as_not_approved():
    rows = desk_science._parse("a\tb\tc\td\te\n5s\t15s\t20d\t14n\t10s\n"
                               "USGS\t1\t2020-01-01\tIce\tA\nUSGS\t1\t2020-01-02\t5\tP\nUSGS\t1\t2020-01-03\t7\tA:e\n")
    assert [(r[1], r[2]) for r in rows] == [(None, "A"), (5.0, "P"), (7.0, "A:e")]
    assert desk_science._low_flow({}, 3) is None


def test_kev_median_allows_rounding_and_rows_must_keep_rank_order():
    task = _task("qe_skill-security-d4", "kev", 2)
    header, *body = skills.solve(task).splitlines()
    cells = body[0].split(",")
    moved = ",".join([*cells[:4], f"{float(cells[4]) + 0.04:.2f}", cells[5]])
    assert skills.grade(task, "\n".join([header, moved, *body[1:]]) + "\n") == 1.0
    moved = ",".join([*cells[:4], f"{float(cells[4]) + 0.5:.1f}", cells[5]])
    assert skills.grade(task, "\n".join([header, moved, *body[1:]]) + "\n") == 0.0
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 0.0


def test_kev_vendor_names_are_trimmed_and_cwe_ties_go_to_the_lowest_number():
    assert sorted(["CWE-119", "CWE-20", "CWE-NVD"], key=desk_security._cwe_number) == ["CWE-20", "CWE-119", "CWE-NVD"]
    assert desk_security._cwes("CWE-79, CWE-89,, NVD-CWE-noinfo") == ["CWE-79", "CWE-89"]
    names = {e["vendorProject"].strip() for e in desk_security._entries(desk_security._snapshot())}
    assert all(name == name.strip() for name in names)


def test_yields_reads_files_as_published_and_spells_cross_years():
    years = {}
    for seed in range(60):
        task = _task("qe_skill-finance-d4", "yields", seed)
        csvs = sorted(n for n in task.files if n.endswith(".csv"))
        years[len(csvs)] = True
        first = desk_finance._days(task.files[csvs[0]])
        assert first[0]["Date"] > first[-1]["Date"] or first[0]["Date"][-4:] == first[-1]["Date"][-4:]
        for row in task.private["reference"]:
            assert row["start"] <= row["end"] and int(row["trading_days"]) >= 1
    assert set(years) == {2, 3}


def test_yields_spread_is_whole_basis_points_and_zero_is_not_inverted():
    task = _task("qe_skill-finance-d4", "yields", 4)
    header, first, *rest = skills.solve(task).splitlines()
    cells = first.split(",")
    decimal = ",".join([*cells[:3], cells[3] + ".0", cells[4]])
    assert skills.grade(task, "\n".join([header, decimal, *rest]) + "\n") == 1.0
    moved = ",".join([*cells[:3], str(int(cells[3]) - 1), cells[4]])
    assert skills.grade(task, "\n".join([header, moved, *rest]) + "\n") == 0.0
    files = {"daily_treasury_par_yield_curve_2020.csv": 'Date,"10 Yr","2 Yr"\n01/03/2020,1.50,1.60\n'
             "01/02/2020,1.50,1.50\n01/01/2020,1.40,1.50\n",
             "request.json": json.dumps({"long_tenor": "10 Yr", "short_tenor": "2 Yr", "min_spell_days": 1})}
    assert desk_finance._yields_solve(files, 4, None).splitlines()[1:] == [
        "2020-01-01,2020-01-01,1,-10,2020-01-01", "2020-01-03,2020-01-03,1,-10,2020-01-03"]


def test_ledger_reads_every_notation_exactly():
    cases = {"12": 12, "-3.75": Fraction(-15, 4), "7/12": Fraction(7, 12), "2 3/4": Fraction(11, 4),
             "-2 3/4": Fraction(-11, 4), "0.1(6)": Fraction(1, 6), "1.(142857)": Fraction(8, 7),
             "-0.(3)": Fraction(-1, 3), "125e-3": Fraction(1, 8)}
    for text, value in cases.items():
        assert desk_math._parse(text) == value, text
    assert desk_math._format(Fraction(-17, 6)) == "-17/6" and desk_math._format(Fraction(4)) == "4"


def test_ledger_totals_must_be_exact_reduced_strings():
    task = _task("qe_skill-math-d4", "ledger", 1)
    header, first, *rest = skills.solve(task).splitlines()
    group, entries, total = first.split(",")
    value = Fraction(total)
    for text in (f"{value.numerator * 2}/{value.denominator * 2}", repr(float(value)), f"+{total}"):
        assert skills.grade(task, "\n".join([header, f"{group},{entries},{text}", *rest]) + "\n") == 0.0, text
