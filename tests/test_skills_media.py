"""media family specifics: drop-frame arithmetic against SMPTE reference points, and the output variations the
checks accept (formatting) or reject (a near miss). The generic oracle and trap tests live in test_skills.py."""

from __future__ import annotations

import json

import pytest

from qe_arena import skills
from qe_arena.skills.media import _iso_duration, _tc_frame, _tc_label

SEEDS = range(60)


def _task(kind: str, level: int, seed: int) -> skills.SkillTask:
    return skills.make_skill_task(f"qe_skill-media-d{level}", seed, pool=(kind,))


@pytest.mark.parametrize(("label", "frame"), [("00:00:59;29", 1799), ("00:01:00;02", 1800), ("00:02:00;02", 3598),
                                              ("00:10:00;00", 17982), ("00:11:00;02", 19782),
                                              ("01:00:00;00", 107892), ("23:59:59;29", 2589407)])
def test_drop_frame_timecode_matches_smpte_reference_points(label, frame):
    assert _tc_frame(label, 30, True) == frame
    assert _tc_label(frame, 30, True) == label


def test_drop_frame_round_trips_and_never_emits_a_skipped_label():
    for frame in range(0, 200_000, 3):
        label = _tc_label(frame, 30, True)
        assert _tc_frame(label, 30, True) == frame
        minute, ff = int(label[3:5]), int(label[-2:])
        assert not (label[6:8] == "00" and ff < 2 and minute % 10), label


def test_iso_durations_drop_zero_components():
    assert [_iso_duration(s) for s in (3725, 3600, 300, 45, 7201)] == ["PT1H2M5S", "PT1H", "PT5M", "PT45S", "PT2H1S"]


@pytest.mark.parametrize("level", (1, 2, 3))
def test_json_outputs_accept_reformatting_and_reject_near_misses(level):
    for seed in SEEDS:
        task = _task("timecode", level, seed)
        out = json.loads(skills.solve(task))
        assert skills.grade(task, json.dumps(out, indent=4) + "\n") == 1.0
        first = next(iter(out["clips"]))
        off_by_one = dict(out["clips"], **{first: out["clips"][first] + 1})
        assert skills.grade(task, json.dumps(dict(out, clips=off_by_one))) == 0
        assert skills.grade(task, json.dumps(dict(out, total_frames=float(out["total_frames"])))) == 0
        other = out["total_timecode"].replace(";", ":") if level == 3 else out["total_timecode"][:8] + ";" + out[
            "total_timecode"][9:]
        assert skills.grade(task, json.dumps(dict(out, total_timecode=other))) == 0

        task = _task("catalogue", level, seed)
        out = json.loads(skills.solve(task))
        assert skills.grade(task, json.dumps(out, indent=2)) == 1.0
        assert skills.grade(task, json.dumps(dict(out, items=out["items"][::-1]))) == 0  # order is stated
        padded = [dict(out["items"][0], title=" " + out["items"][0]["title"]), *out["items"][1:]]
        assert skills.grade(task, json.dumps(dict(out, items=padded))) == 0
        assert skills.grade(task, json.dumps(dict(out, extra=1))) == 0


@pytest.mark.parametrize("level", (1, 2, 3))
def test_crop_csv_accepts_a_missing_final_newline_and_rejects_reordered_or_float_rows(level):
    for seed in SEEDS:
        task = _task("assets", level, seed)
        text = skills.solve(task)
        assert skills.grade(task, text.rstrip("\n")) == 1.0
        assert skills.grade(task, text.replace("\n", "\r\n")) == 1.0
        header, *body = text.strip().split("\n")
        if len(body) > 1:
            assert skills.grade(task, "\n".join([header, *body[::-1]]) + "\n") == 0
        first = body[0].split(",")
        assert skills.grade(task, "\n".join([header, ",".join([*first[:-1], first[-1] + ".0"]), *body[1:]])) == 0
