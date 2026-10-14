"""Convert a non-drop-frame SMPTE timecode between integer frame rates. Original."""

import re

META = {
    "entry": "convert_timecode",
    "origin": "Original.",
    "tags": ["timecode", "smpte", "frame-rate"],
    "domain": "media-content-production",
}


def convert_timecode(timecode, src_fps, dst_fps):
    """Convert a non-drop-frame "HH:MM:SS:FF" timecode; return [dst frames, dst timecode].

    Rates must be 24, 25, 30, 48, 50 or 60. Fields are two digits: HH 00-23, MM and SS
    00-59, FF 00 to src_fps - 1. frames = ((HH * 60 + MM) * 60 + SS) * src_fps + FF becomes
    frames * dst_fps // src_fps (rounded down) at dst_fps, written back as "HH:MM:SS:FF".
    Raises ValueError for another rate, a malformed timecode or a field out of range.
    """
    for fps in (src_fps, dst_fps):
        if fps not in (24, 25, 30, 48, 50, 60):
            raise ValueError("unsupported frame rate")
    match = re.fullmatch(r"(\d\d):(\d\d):(\d\d):(\d\d)", timecode)
    if not match:
        raise ValueError("timecode must be HH:MM:SS:FF")
    hours, minutes, seconds, frame = (int(part) for part in match.groups())
    if hours > 23 or minutes > 59 or seconds > 59 or frame >= src_fps:
        raise ValueError("field out of range")
    count = ((hours * 60 + minutes) * 60 + seconds) * src_fps + frame
    frames = count * dst_fps // src_fps
    whole, frame = divmod(frames, dst_fps)
    return [frames, "%02d:%02d:%02d:%02d" % (whole // 3600, whole // 60 % 60, whole % 60, frame)]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["00:00:00:00", 24, 25],
        ["01:00:00:00", 25, 30],
        ["00:00:01:23", 24, 48],
        ["00:00:01:24", 24, 48],
        ["00:00:01:29", 30, 25],
        ["00:00:01:12", 25, 24],
        ["23:59:59:59", 60, 50],
        ["24:00:00:00", 25, 25],
        ["00:60:00:00", 25, 25],
        ["00:00:60:00", 25, 25],
        ["00:59:59:24", 25, 60],
        ["10:10:10:10", 23, 25],
        ["10:10:10:10", 25, 29],
        ["1:00:00:00", 25, 25],
        ["01:00:00;00", 30, 30],
        ["12:34:56:07", 30, 24],
    ]
    rates = [24, 25, 30, 48, 50, 60]
    while len(cases) < n:
        src, dst = rng.choice(rates), rng.choice(rates + [29] if rng.random() < 0.1 else rates)
        hours = rng.choice([0, 0, 0, 1, rng.randint(0, 23)])  # most clips are under an hour
        last_frame = src - 1 if rng.random() < 0.9 else src  # sometimes one frame too many
        parts = [hours, rng.choice([0, rng.randint(0, 59)]), rng.randint(0, 59), rng.randint(0, last_frame)]
        cases.append([":".join("%02d" % p for p in parts), src, dst])
    return cases[:n]
