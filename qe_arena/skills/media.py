"""media-content-production families for qe_skill."""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction

from . import Built, Family

DOMAIN = "media-content-production"

# D1 subtitles: retime and repair an SRT file to an edit sheet.
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


def _render(cues: list[tuple[int, int, int, list[str]]]) -> str:
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


def _subtitles_build(level: int, rng: random.Random) -> Built:
    count = rng.randint(6, 10)
    job: dict = {"shift_ms": rng.choice((-1, 1)) * rng.randint(400, 4000)}
    if level >= 2:
        job.update({"source_fps": "23.976", "target_fps": "25"})
    if level == 3:
        job.update({"min_gap_ms": 80, "max_line_chars": 42})
    t = rng.randint(6000, 20000)
    cues = []
    for i in range(count):
        duration = rng.randint(1200, 4200)
        lines = [_sentence(rng, rng.randint(3, 6))]
        if level == 3 and i == 1:
            lines = [_sentence(rng, 11)]  # at least one line longer than 42 characters
        cues.append([i + 1, t, t + duration, lines])
        t += duration + rng.randint(30, 900)
    if level == 3:  # at least one pair closer than the minimum gap
        k = rng.randrange(count - 1)
        cues[k][2] = cues[k + 1][1] + rng.randint(-60, 40)
        order = list(range(count))
        rng.shuffle(order)
        first = rng.randint(1, 20)
        cues = [[first + 2 * n, *cues[i][1:]] for n, i in enumerate(order)]
    files = {"input.srt": _render([tuple(c) for c in cues]), "job.json": json.dumps(job, indent=2) + "\n"}
    rules = ["Shift every start and end time by shift_ms milliseconds (negative means earlier)."]
    if level >= 2:
        rules = ["The cues were timed against source_fps video; the video now runs at target_fps (PAL speed-up), "
                 "so first rescale every time t to t x source_fps / target_fps, then apply shift_ms."]
    if level == 3:
        rules += ["The cues in input.srt are out of order and misnumbered: sort them by start time and number them "
                  "1, 2, 3, ...",
                  "After retiming, no cue may end later than min_gap_ms before the next cue starts; where one does, "
                  "move its end time earlier to exactly that point.",
                  "No text line may be longer than max_line_chars characters: break a longer line at the last "
                  "space that keeps the first part within the limit, and repeat for the rest."]
    statement = "\n".join([
        "input.srt is a subtitle file and job.json is the edit sheet for it.",
        *rules,
        "Round every time to the nearest millisecond. Keep the text otherwise unchanged.",
        "Write the result to output.srt in standard SRT format (index line, 'HH:MM:SS,mmm --> HH:MM:SS,mmm' line, "
        "text lines, blank line between cues).",
    ])
    built = Built(files=files, statement=statement, output="output.srt", reference=None)
    built.reference = _parse(_subtitles_solve(files, level, None))
    return built


def _subtitles_solve(files: dict[str, str], level: int, variant: str | None) -> str:
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
    if level == 3:
        if variant != "no_sort":
            out = [(n + 1, s, e, lines) for n, (_, s, e, lines) in enumerate(sorted(out, key=lambda c: c[1]))]
        if variant != "no_gap":
            gap = job["min_gap_ms"]
            out = [(i, s, min(e, out[k + 1][1] - gap) if k + 1 < len(out) else e, lines)
                   for k, (i, s, e, lines) in enumerate(out)]
        if variant != "no_wrap":
            width = job["max_line_chars"]
            out = [(i, s, e, [part for line in lines for part in _wrap(line, width)]) for i, s, e, lines in out]
    return _render(out)


def _subtitles_check(reference: list, text: str) -> list[bool]:
    cues = _parse(text)
    results = [len(cues) == len(reference)]
    if not results[0]:
        return results
    for (idx, start, end, lines), (r_idx, r_start, r_end, r_lines) in zip(cues, reference):
        results += [idx == r_idx, abs(start - r_start) <= TOLERANCE_MS, abs(end - r_end) <= TOLERANCE_MS,
                    lines == r_lines]
    return results


FAMILIES = (
    Family(name="subtitles", domain=DOMAIN, build=_subtitles_build, check=_subtitles_check, solve=_subtitles_solve,
           traps={"wrong_sign": (1, 2, 3), "no_rate": (2, 3), "inverted_rate": (2, 3), "shift_first": (2, 3),
                  "no_sort": (3,), "no_gap": (3,), "no_wrap": (3,)}),
)
