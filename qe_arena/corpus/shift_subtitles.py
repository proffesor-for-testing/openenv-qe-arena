"""Shift SRT subtitle cue times by an offset and flag cues that overlap. Original."""

import re

META = {
    "entry": "shift_subtitles",
    "origin": "Original.",
    "tags": ["subtitles", "srt", "timecode"],
    "domain": "media-content-production",
}


def srt_ms(stamp):
    """Milliseconds for an SRT timestamp "HH:MM:SS,mmm"; ValueError when malformed."""
    match = re.fullmatch(r"(\d\d):([0-5]\d):([0-5]\d),(\d\d\d)", stamp)
    if not match:
        raise ValueError("bad SRT timestamp: " + stamp)
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def srt_stamp(ms):
    """The SRT timestamp "HH:MM:SS,mmm" for 0 <= ms <= 359999999."""
    seconds, millis = divmod(ms, 1000)
    hours, seconds = divmod(seconds, 3600)
    return "%02d:%02d:%02d,%03d" % (hours, seconds // 60, seconds % 60, millis)


def shift_subtitles(cues, offset_ms):
    """Shift every cue by offset_ms (int, may be negative) and report overlapping cues.

    cues is a list of [start, end] SRT timestamps "HH:MM:SS,mmm" (two-digit hours, minutes
    and seconds 00-59, three-digit milliseconds). After shifting, a cue whose end is at or
    before 0 ms is dropped, and a kept cue whose start is negative starts at 00:00:00,000.
    Returns {"cues": the kept cues as [start, end] timestamps in input order, "overlaps": the
    kept-list positions of cues that start before the previous kept cue ends (not at its end)}.
    Raises ValueError for a malformed timestamp, a cue whose end is not after its start, or
    a kept cue whose shifted end is past 99:59:59,999.
    """
    kept, overlaps, previous_end = [], [], None
    for start_stamp, end_stamp in cues:
        start, end = srt_ms(start_stamp), srt_ms(end_stamp)
        if end <= start:
            raise ValueError("cue ends before it starts: " + start_stamp)
        start, end = max(0, start + offset_ms), end + offset_ms
        if end <= 0:
            continue
        if end > 359999999:
            raise ValueError("shifted cue ends past 99:59:59,999")
        if kept and start < previous_end:
            overlaps.append(len(kept))
        kept.append([srt_stamp(start), srt_stamp(end)])
        previous_end = end
    return {"cues": kept, "overlaps": overlaps}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[], 0],
        [[["00:00:01,000", "00:00:02,500"], ["00:00:02,500", "00:00:04,000"]], 0],
        [[["00:00:01,000", "00:00:02,500"], ["00:00:02,499", "00:00:04,000"]], 250],
        [[["00:00:01,000", "00:00:02,000"], ["00:00:03,000", "00:00:04,000"]], -2000],
        [[["00:00:01,000", "00:00:02,000"], ["00:00:03,000", "00:00:04,000"]], -1999],
        [[["00:00:00,500", "00:00:01,200"]], -600],
        [[["01:59:59,999", "02:00:00,000"]], 1],
        [[["99:59:58,000", "99:59:59,000"]], 999],
        [[["99:59:58,000", "99:59:59,000"]], 1000],
        [[["00:00:05,000", "00:00:05,000"]], 0],
        [[["00:00:05,000", "00:00:04,999"]], 0],
        [[["0:00:05,000", "00:00:06,000"]], 0],
        [[["00:60:05,000", "01:00:06,000"]], 0],
        [[["00:00:05.000", "00:00:06,000"]], 0],
        [[["00:00:01,000", "00:10:00,000"], ["00:00:02,000", "00:00:03,000"], ["00:00:04,000", "00:00:05,000"]], 0],
    ]
    while len(cases) < n:
        cues, t = [], rng.randint(0, 3 * 3600000)
        for _ in range(rng.randint(1, 6)):
            t = max(0, t + rng.randint(-800, 4000))
            length = rng.randint(1, 6000)
            cues.append([srt_stamp(t), srt_stamp(t + length)])
            t += length
        cases.append([cues, rng.choice([0, rng.randint(-5000, 5000), rng.randint(-4000000, 4000000)])])
    return cases[:n]
