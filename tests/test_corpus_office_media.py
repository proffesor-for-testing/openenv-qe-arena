"""Spec checks for the v4 office-white-collar and media-content-production corpus entries.

Each docstring is the model's spec, so these tests recompute the entries from that spec with
independent code (datetime, Fraction, brute force) rather than re-reading the implementation.
"""
from __future__ import annotations

import datetime
import json
import random
from fractions import Fraction

import pytest

from qe_arena import corpus
from qe_arena.corpus import call_reference, gen_inputs

OFFICE = ("networkdays", "validate_csv_row", "booking_conflicts", "expense_violations", "mail_merge")
MEDIA = ("shift_subtitles", "select_bitrate_rung", "convert_timecode", "fit_frame", "loudness_gain")
EPOCH = datetime.date(1970, 1, 1)


def _call(name, args):
    try:
        return "ok", call_reference(name, json.loads(json.dumps(args)))
    except (ValueError, TypeError, KeyError) as exc:
        return "raises", type(exc).__name__


@pytest.mark.parametrize("name, domain", [(n, "office-white-collar") for n in OFFICE]
                         + [(n, "media-content-production") for n in MEDIA])
def test_meta_names_its_domain_and_is_original(name, domain):
    meta = corpus._module(name).META
    assert set(meta) == {"entry", "origin", "tags", "domain"}
    assert meta["domain"] == domain and meta["origin"] == "Original."
    assert name in corpus.INJECTED_NAMES


def _brute_networkdays(start, end, holidays):
    lo, hi = min(start, end), max(start, end)
    days = [d for d in range(lo, hi + 1) if (EPOCH + datetime.timedelta(d)).weekday() < 5]
    count = len(set(days) - set(holidays))
    return -count if start > end else count


def test_networkdays_matches_a_calendar_brute_force():
    for seed in range(5):
        for args in gen_inputs("networkdays", seed, 64):
            if all(isinstance(v, int) and not isinstance(v, bool) for v in [args[0], args[1], *args[2]]):
                assert call_reference("networkdays", args) == _brute_networkdays(*args), args
    assert call_reference("networkdays", [20108, 20112, []]) == 5  # Monday 2025-01-20 .. Friday


def test_shift_subtitles_timestamps_round_trip():
    module = corpus._module("shift_subtitles")
    rng = random.Random(4)
    for ms in [0, 999, 1000, 59999, 3599999, 3600000, 359999999] + [rng.randrange(360000000) for _ in range(200)]:
        assert module.srt_ms(module.srt_stamp(ms)) == ms


def test_convert_timecode_matches_exact_fraction_arithmetic():
    for seed in range(5):
        for tc, src, dst in gen_inputs("convert_timecode", seed, 64):
            kind, result = _call("convert_timecode", [tc, src, dst])
            if kind != "ok":
                continue
            h, m, s, f = (int(p) for p in tc.split(":"))
            instant = Fraction((h * 60 + m) * 60 * src + s * src + f, src)
            frames = int(instant * dst)  # floor: instant is non-negative
            assert result["frames"] == frames
            assert result["timecode"] == "%02d:%02d:%02d:%02d" % (
                frames // dst // 3600, frames // dst // 60 % 60, frames // dst % 60, frames % dst)
    assert call_reference("convert_timecode", ["01:00:00:00", 25, 30]) == {"frames": 108000, "timecode": "01:00:00:00"}


def test_fit_frame_boxes_stay_inside_and_keep_the_shape():
    for seed in range(5):
        for sw, sh, dw, dh, mode in gen_inputs("fit_frame", seed, 64):
            kind, result = _call("fit_frame", [sw, sh, dw, dh, mode])
            if kind != "ok":
                continue
            x, y, w, h = result["box"]
            outer_w, outer_h = (dw, dh) if mode == "fit" else (sw, sh)
            assert 0 <= x and 0 <= y and x + w <= outer_w and y + h <= outer_h
            if result["adjust"] == "none":
                assert Fraction(sw, sh) == Fraction(dw, dh) and [x, y] == [0, 0]


def test_booking_conflicts_turnover_examples_from_the_docstring():
    back_to_back = [["Oak", 540, 600], ["Oak", 600, 660]]
    assert call_reference("booking_conflicts", [back_to_back, 0]) == []
    assert call_reference("booking_conflicts", [back_to_back, 10]) == [[0, 1]]
    assert call_reference("booking_conflicts", [[["Oak", 540, 600], ["Oak", 610, 660]], 10]) == []


def test_mail_merge_defaults_and_missing_fields():
    out = call_reference("mail_merge", ["Hi {{ name }}, {{city|HQ}} {{zip}}{{zip}}", {"name": "Ana", "zip": None}])
    assert out == {"text": "Hi Ana, HQ ", "missing": ["zip"]}


def test_loudness_gain_peak_limit_can_turn_a_boost_into_a_cut():
    out = call_reference("loudness_gain", [-900, 150, -1400, 0, 0])
    assert out == {"gain_cdb": -500, "loudness_cdb": -1400, "peak_cdb": -350, "limited_by": "target"}
    out = call_reference("loudness_gain", [-2000, 200, -1400, -100, 1200])
    assert out["limited_by"] == "peak" and out["gain_cdb"] == -300


@pytest.mark.parametrize("name", OFFICE + MEDIA)
def test_edge_cases_cover_both_valid_and_rejected_calls(name):
    kinds = {_call(name, args)[0] for args in gen_inputs(name, 0, 20)}
    assert kinds == {"ok", "raises"}
