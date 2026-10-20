"""The media `subtitles` family: retime and repair an SRT file to an edit sheet.

Recalibrated 2026-10-20: the first version was too easy at every level (14, 13 and 15 of 16 strict solves), so each
level now stacks more of the repair rules, and d3 adds a minimum display duration that interacts with the gap rule.
"""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction

from . import Built, Family

DOMAIN = "media-content-production"
WORDS = ("the", "harbour", "light", "was", "never", "meant", "to", "go", "out", "tonight", "she", "said", "we",
         "have", "until", "morning", "keep", "the", "engine", "running", "and", "do", "not", "look", "back", "at",
         "storm", "coming", "over", "ridge", "listen", "carefully", "this", "is", "last", "train", "north")
TIME = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$")
TOLERANCE_MS = 1


def _stamp(ms: int) -> str:
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, milli = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _render(cues: list) -> str:
    return "\n".join(f"{idx}\n{_stamp(start)} --> {_stamp(end)}\n" + "\n".join(lines) + "\n"
                     for idx, start, end, lines in cues)


def _parse(text: str) -> list[tuple[int, int, int, list[str]]]:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    cues = []
    for block in blocks:
        lines = [line.rstrip() for line in block.split("\n")]
        match = TIME.match(lines[1].strip())
        g = [int(v) for v in match.groups()]
        start = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
        end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
        if not lines[2:]:
            raise ValueError("cue without text")
        cues.append((int(lines[0].strip()), start, end, lines[2:]))
    return cues


def _sentence(rng: random.Random, words: int) -> str:
    text = " ".join(rng.choice(WORDS) for _ in range(words))
    return text[0].upper() + text[1:] + rng.choice((".", "?", "!", "..."))


def _wrap(line: str, width: int) -> list[str]:
    out = []
    while len(line) > width:
        cut = line.rfind(" ", 0, width + 1)
        if cut <= 0:
            break
        out.append(line[:cut])
        line = line[cut + 1:]
    return out + [line]


def _build(level: int, rng: random.Random) -> Built:
    count = {1: rng.randint(8, 11), 2: rng.randint(10, 13), 3: rng.randint(12, 16)}[level]
    job: dict = {"shift_ms": rng.choice((-1, 1)) * rng.randint(400, 4000)}
    if level >= 2:
        job.update({"source_fps": "23.976", "target_fps": "25", "min_gap_ms": 80})
    if level == 3:
        job.update({"max_line_chars": 42, "min_duration_ms": 1000})
    t = rng.randint(6000, 20000)
    cues = []
    long_lines = set(rng.sample(range(count), 2)) if level == 3 else set()
    short = set(rng.sample(range(count - 1), 3)) if level == 3 else set()
    for i in range(count):
        duration = rng.randint(450, 850) if i in short else rng.randint(1300, 4200)
        lines = [_sentence(rng, 11 if i in long_lines else rng.randint(3, 6))]
        cues.append([0, t, t + duration, lines])
        t += duration + rng.randint(30, 900)
    if level >= 2:  # pairs closer than the minimum gap, one of them right after a short cue at d3
        for k in rng.sample(range(count - 1), 2):
            cues[k][2] = cues[k + 1][1] + rng.randint(-60, 40)
        for k in short:
            cues[k + 1][1] = cues[k][1] + rng.randint(900, 1100)
            cues[k + 1][2] = max(cues[k + 1][2], cues[k + 1][1] + 1300)
    order = list(range(count))
    rng.shuffle(order)
    first = rng.randint(1, 20)
    cues = [[first + 2 * n, *cues[i][1:]] for n, i in enumerate(order)]
    files = {"input.srt": _render(cues), "job.json": json.dumps(job, indent=2) + "\n"}
    rules = ["The cues in input.srt are out of order and misnumbered: sort them by start time and number them "
             "1, 2, 3, ..."]
    rules.append("Shift every start and end time by shift_ms milliseconds (negative means earlier)." if level == 1 else
                 "The cues were timed against source_fps video; the video now runs at target_fps (PAL speed-up), "
                 "so first rescale every time t to t x source_fps / target_fps, then apply shift_ms.")
    if level == 3:
        rules.append("After retiming, any cue shorter than min_duration_ms gets its end moved to start + "
                     "min_duration_ms. Apply this before the gap rule.")
    if level >= 2:
        rules.append("Then no cue may end later than min_gap_ms before the next cue starts; where one does, move its end "
                     "time earlier to exactly that point (the gap rule wins over the minimum duration).")
    if level == 3:
        rules.append("No text line may be longer than max_line_chars characters: break a longer line at the last "
                     "space that keeps the first part within the limit, and repeat for the rest.")
    statement = "\n".join([
        "input.srt is a subtitle file and job.json is the edit sheet for it.",
        *rules,
        "Round every time to the nearest millisecond. Keep the text otherwise unchanged.",
        "Write the result to output.srt in standard SRT format (index line, 'HH:MM:SS,mmm --> HH:MM:SS,mmm' line, "
        "text lines, blank line between cues).",
    ])
    built = Built(files=files, statement=statement, output="output.srt", reference=None)
    built.reference = _parse(_solve(files, level, None))
    return built


def _solve(files: dict[str, str], level: int, variant: str | None) -> str:
    job = json.loads(files["job.json"])
    cues = _parse(files["input.srt"])
    shift = -job["shift_ms"] if variant == "wrong_sign" else job["shift_ms"]
    ratio = Fraction(1)
    if level >= 2 and variant != "no_rate":
        ratio = Fraction(job["source_fps"]) / Fraction(job["target_fps"])
        if variant == "inverted_rate":
            ratio = 1 / ratio

    def retime(ms: int) -> int:
        if variant == "shift_first":
            return round((ms + shift) * ratio)
        return round(ms * ratio) + shift

    out = [(idx, retime(start), retime(end), lines) for idx, start, end, lines in cues]
    if variant != "no_sort":
        out = [(n + 1, s, e, lines) for n, (_, s, e, lines) in enumerate(sorted(out, key=lambda c: c[1]))]

    def min_duration(cues: list) -> list:
        least = job["min_duration_ms"]
        return [(i, s, max(e, s + least), lines) for i, s, e, lines in cues]

    def gap(cues: list) -> list:
        g = job["min_gap_ms"]
        return [(i, s, min(e, cues[k + 1][1] - g) if k + 1 < len(cues) else e, lines)
                for k, (i, s, e, lines) in enumerate(cues)]

    if level == 3 and variant not in ("no_min_duration", "gap_before_duration"):
        out = min_duration(out)
    if level >= 2 and variant != "no_gap":
        out = gap(out)
    if level == 3 and variant == "gap_before_duration":
        out = min_duration(out)
    if level == 3 and variant != "no_wrap":
        width = job["max_line_chars"]
        out = [(i, s, e, [part for line in lines for part in _wrap(line, width)]) for i, s, e, lines in out]
    return _render(out)


def _check(reference: list, text: str) -> list[bool]:
    cues = _parse(text)
    results = [len(cues) == len(reference)]
    if not results[0]:
        return results
    for (idx, start, end, lines), (r_idx, r_start, r_end, r_lines) in zip(cues, reference):
        results += [idx == r_idx, abs(start - r_start) <= TOLERANCE_MS, abs(end - r_end) <= TOLERANCE_MS,
                    lines == r_lines]
    return results


SUBTITLES = Family(name="subtitles", domain=DOMAIN, build=_build, check=_check, solve=_solve,
                   traps={"wrong_sign": (1, 2, 3), "no_sort": (1, 2, 3), "no_rate": (2, 3), "inverted_rate": (2, 3),
                          "shift_first": (2, 3), "no_gap": (2, 3), "no_wrap": (3,), "no_min_duration": (3,),
                          "gap_before_duration": (3,)})
