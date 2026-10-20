"""Family-specific checks for the natural-science qe_skill families: stated tolerances and output-shape rules."""

from __future__ import annotations

import json

import pytest

from qe_arena import skills


def _task(level: int, kind: str, seed: int = 0) -> skills.SkillTask:
    return skills.make_skill_task(f"qe_skill-science-d{level}", seed, pool=(kind,))


@pytest.mark.parametrize("level", (1, 2, 3))
def test_calibration_accepts_its_stated_rounding_and_rejects_more(level):
    task = _task(level, "calibration")
    out = json.loads(skills.solve(task))
    sample = next(iter(out["concentrations"]))
    for delta, expected in ((0.0004, 1.0), (-0.0004, 1.0), (0.002, 0.0)):
        moved = dict(out, concentrations=dict(out["concentrations"], **{sample: out["concentrations"][sample] + delta}))
        assert skills.grade(task, json.dumps(moved)) == expected, delta
    for scale, expected in ((1 + 4e-6, 1.0), (1 + 1e-4, 0.0)):
        assert skills.grade(task, json.dumps(dict(out, slope=out["slope"] * scale))) == expected, scale
    assert skills.grade(task, json.dumps(dict(out, r_squared=True))) == 0.0


@pytest.mark.parametrize("level", (1, 2, 3))
def test_decay_accepts_one_percent_and_rejects_two(level):
    task = _task(level, "decay")
    out = json.loads(skills.solve(task))
    for scale, expected in ((1.009, 1.0), (0.991, 1.0), (1.02, 0.0)):
        assert skills.grade(task, json.dumps(dict(out, half_life_min=out["half_life_min"] * scale))) == expected


def test_dilution_targets_may_come_in_any_order_but_serial_rows_may_not():
    task = _task(1, "dilution")
    header, *body = skills.solve(task).splitlines()
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 1.0
    assert skills.grade(task, "\n".join([header, *body[:-1]]) + "\n") == 0.0
    nan = body[0].rsplit(",", 1)[0] + ",nan"
    assert skills.grade(task, "\n".join([header, nan, *body[1:]]) + "\n") == 0.0
    task = _task(2, "dilution")
    header, *body = skills.solve(task).splitlines()
    assert skills.grade(task, "\n".join([header, *reversed(body)]) + "\n") == 0.0


def test_dilution_volumes_allow_the_rounding_unit_only():
    task = _task(2, "dilution")
    header, first, *rest = skills.solve(task).splitlines()
    cells = first.split(",")
    for delta, expected in ((0.04, 1.0), (0.2, 0.0)):
        moved = ",".join([*cells[:2], f"{float(cells[2]) + delta:.2f}", *cells[3:]])
        assert skills.grade(task, "\n".join([header, moved, *rest]) + "\n") == expected, delta


def test_every_level_three_dilution_needs_an_intermediate_tube():
    for seed in range(100):
        task = _task(3, "dilution", seed)
        assert "-I1," in skills.solve(task), seed
