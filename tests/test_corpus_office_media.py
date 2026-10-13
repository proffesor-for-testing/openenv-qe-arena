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
MEDIA = ("shift_cue", "select_bitrate_rung", "convert_timecode", "crop_to_aspect", "loudness_gain")
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


def test_shift_cue_matches_millisecond_arithmetic():
    def ms(stamp):
        h, m, rest = stamp.split(":")
        sec, milli = rest.split(",")
        return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(milli)

    for seed in range(5):
        for start, end, offset in gen_inputs("shift_cue", seed, 64):
            kind, result = _call("shift_cue", [start, end, offset])
            if kind != "ok":
                continue
            if result is None:
                assert ms(end) + offset <= 0
            else:
                assert ms(result[1]) == ms(end) + offset and ms(result[0]) == max(0, ms(start) + offset)
    assert call_reference("shift_cue", ["00:59:59,500", "01:00:01,000", 600]) == ["01:00:00,100", "01:00:01,600"]


def test_convert_timecode_matches_exact_fraction_arithmetic():
    for seed in range(5):
        for tc, src, dst in gen_inputs("convert_timecode", seed, 64):
            kind, result = _call("convert_timecode", [tc, src, dst])
            if kind != "ok":
                continue
            h, m, s, f = (int(p) for p in tc.split(":"))
            instant = Fraction((h * 60 + m) * 60 * src + s * src + f, src)
            frames = int(instant * dst)  # floor: instant is non-negative
            assert result[0] == frames
            assert result[1] == "%02d:%02d:%02d:%02d" % (
                frames // dst // 3600, frames // dst // 60 % 60, frames // dst % 60, frames % dst)
    assert call_reference("convert_timecode", ["01:00:00:00", 25, 30]) == [108000, "01:00:00:00"]


def test_crop_to_aspect_boxes_stay_inside_and_match_the_aspect():
    for seed in range(5):
        for sw, sh, aw, ah in gen_inputs("crop_to_aspect", seed, 64):
            kind, result = _call("crop_to_aspect", [sw, sh, aw, ah])
            if kind != "ok":
                continue
            x, y, w, h, cut = result
            assert 0 <= x and 0 <= y and x + w <= sw and y + h <= sh
            assert {"width": h == sh, "height": w == sw, "none": [w, h] == [sw, sh]}[cut]
            # the kept box is the largest one with at most the target aspect error of one pixel
            assert abs(Fraction(w * ah - aw * h)) < max(aw, ah) or cut == "none"
    assert call_reference("crop_to_aspect", [1920, 1080, 4, 3]) == [240, 0, 1440, 1080, "width"]


def test_booking_conflicts_turnover_examples_from_the_docstring():
    back_to_back = [["Oak", 540, 600], ["Oak", 600, 660]]
    assert call_reference("booking_conflicts", [back_to_back, 0]) == []
    assert call_reference("booking_conflicts", [back_to_back, 10]) == [[0, 1, 10]]
    assert call_reference("booking_conflicts", [[["Oak", 540, 600], ["Oak", 610, 660]], 10]) == []


def test_mail_merge_defaults_and_missing_fields():
    out = call_reference("mail_merge", ["Hi {{ name }}, {{city|HQ}} {{zip}}{{zip}}", {"name": "Ana", "zip": None}])
    assert out == ["Hi Ana, HQ ", ["zip"]]


def test_loudness_gain_peak_limit_can_turn_a_boost_into_a_cut():
    assert call_reference("loudness_gain", [-900, 150, -1400, 0, 0]) == [-500, "target"]
    assert call_reference("loudness_gain", [-2000, 200, -1400, -100, 1200]) == [-300, "peak"]


@pytest.mark.parametrize("name", OFFICE + MEDIA)
def test_edge_cases_cover_both_valid_and_rejected_calls(name):
    kinds = {_call(name, args)[0] for args in gen_inputs(name, 0, 20)}
    assert kinds == {"ok", "raises"}
