"""Shift one SRT subtitle cue by a timing offset, clamping or dropping it at zero. Original."""

import re

META = {
    "entry": "shift_cue",
    "origin": "Original.",
    "tags": ["subtitles", "srt", "timecode"],
    "domain": "media-content-production",
}


def shift_cue(start, end, offset_ms):
    """Shift an SRT cue by offset_ms (int, may be negative); return [start, end] or None.

    start and end are "HH:MM:SS,mmm" (two-digit hours, minutes and seconds 00-59, three-digit
    milliseconds). A cue whose shifted end is at or before 0 ms is dropped (None); a shifted
    start below 0 becomes "00:00:00,000". Raises ValueError for a malformed timestamp, an end
    not after the start, or a shifted end past 99:59:59,999.
    """
    times = []
    for stamp in (start, end):
        match = re.fullmatch(r"(\d\d):([0-5]\d):([0-5]\d),(\d\d\d)", stamp)
        if not match:
            raise ValueError("bad SRT timestamp")
        hours, minutes, seconds, millis = (int(part) for part in match.groups())
        times.append(((hours * 60 + minutes) * 60 + seconds) * 1000 + millis + offset_ms)
    first, last = times
    if last <= first:
        raise ValueError("cue must end after it starts")
    if last <= 0:
        return None
    if last > 359999999:
        raise ValueError("shifted past 99:59:59,999")
    shifted = []
    for ms in (max(0, first), last):
        seconds, millis = divmod(ms, 1000)
        hours, seconds = divmod(seconds, 3600)
        shifted.append("%02d:%02d:%02d,%03d" % (hours, seconds // 60, seconds % 60, millis))
    return shifted


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        ["00:00:01,000", "00:00:02,500", 0],
        ["00:00:01,000", "00:00:02,500", 250],
        ["00:00:01,000", "00:00:02,000", -2000],
        ["00:00:01,000", "00:00:02,000", -1999],
        ["00:00:00,500", "00:00:01,200", -600],
        ["01:59:59,999", "02:00:00,000", 1],
        ["99:59:58,000", "99:59:59,000", 999],
        ["99:59:58,000", "99:59:59,000", 1000],
        ["00:00:05,000", "00:00:05,000", 0],
        ["00:00:05,000", "00:00:04,999", 0],
        ["0:00:05,000", "00:00:06,000", 0],
        ["00:60:05,000", "01:00:06,000", 0],
        ["00:00:05.000", "00:00:06,000", 0],
        ["00:59:59,500", "01:00:01,000", 600],
    ]
    while len(cases) < n:
        t = rng.choice([rng.randint(0, 20000), rng.randint(0, 3 * 3600000)])
        length = rng.randint(1, 6000) if rng.random() < 0.92 else -rng.randint(0, 500)
        start, end = [shift_cue("00:00:00,000", "00:00:00,001", ms) for ms in (t, max(0, t + length))]
        cases.append([start[0], end[0], rng.choice([0, rng.randint(-5000, 5000), rng.randint(-4000000, 4000000)])])
    return cases[:n]
