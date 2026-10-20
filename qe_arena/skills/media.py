"""media-content-production families for qe_skill."""

from __future__ import annotations

import json
import random
import re
from datetime import date, timedelta
from fractions import Fraction

from . import Built, Family
from ._common import load_object, rows, write_csv

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


# D2 timecode: per-clip frames and total runtime from an edit decision list.
NOMINAL = {"24": 24, "25": 25, "30": 30, "29.97": 30}  # frames counted per timecode second
WRONG_FPS = {"24": "30", "25": "30", "30": "25"}
DF_TEN_MIN, DF_MIN = 17982, 1798  # 29.97 drop-frame: frames per 10 minutes and per dropped minute
TC = re.compile(r"^(\d{2}):(\d{2}):(\d{2})([:;])(\d{2})$")


def _tc_label(frame: int, fps: int, drop: bool) -> str:
    if drop:  # skip labels ;00 and ;01 at every minute not divisible by 10
        tens, rest = divmod(frame, DF_TEN_MIN)
        frame += 18 * tens + (2 * ((rest - 2) // DF_MIN) if rest > 1 else 0)
    seconds, ff = divmod(frame, fps)
    minutes, ss = divmod(seconds, 60)
    hh, mm = divmod(minutes, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}{';' if drop else ':'}{ff:02d}"


def _tc_frame(label: str, fps: int, drop: bool) -> int:
    hh, mm, ss, _, ff = TC.match(label.strip()).groups()
    total_minutes = int(hh) * 60 + int(mm)
    frame = (total_minutes * 60 + int(ss)) * fps + int(ff)
    return frame - 2 * (total_minutes - total_minutes // 10) if drop else frame


def _timecode_build(level: int, rng: random.Random) -> Built:
    rate = {1: "25", 2: rng.choice(("30", "24")), 3: "29.97"}[level]
    fps, drop = NOMINAL[rate], level == 3
    out_point = rng.choice(("exclusive", "inclusive"))
    records = []
    for n, reel in enumerate(rng.sample(range(1, 90), rng.randint(4, 7))):
        start = rng.randint(0, 3000 * fps)
        length = rng.randint(20 * fps, 200 * fps) if drop else rng.randint(2 * fps, 90 * fps)
        end = start + length - (out_point == "inclusive")
        records.append([f"R{reel:02d}_C{n + 1:02d}", _tc_label(start, fps, drop), _tc_label(end, fps, drop)])
    job = {"fps": rate, "drop_frame": drop, "out_point": out_point}
    files = {"edl.csv": write_csv(["clip", "source_in", "source_out"], records),
             "job.json": json.dumps(job, indent=2) + "\n"}
    notation = "HH:MM:SS;FF" if drop else "HH:MM:SS:FF"
    lines = [
        "edl.csv is an edit decision list: one row per clip with its source in and out points as SMPTE timecode. "
        "job.json gives the frame rate (fps), whether the timecode is drop-frame, and out_point: \"exclusive\" means "
        "source_out is the first frame after the clip, \"inclusive\" means it is the clip's last frame.",
        "Compute each clip's duration in frames, then the total runtime of all clips in frames, and write the total "
        f"as a timecode at the same rate in {notation} notation, counting from {_tc_label(0, fps, drop)}.",
    ]
    if drop:
        lines.insert(1, "The timecode is 29.97 fps drop-frame (HH:MM:SS;FF): frame labels 00 and 01 are skipped at "
                        "the start of every minute except minutes 00, 10, 20, 30, 40 and 50. No frames are dropped, "
                        "only labels.")
    lines.append('Write runtime.json as {"clips": {"<clip>": <frames>, ...}, "total_frames": <frames>, '
                 '"total_timecode": "<timecode>"} with integer frame counts.')
    built = Built(files=files, statement="\n".join(lines), output="runtime.json", reference=None)
    built.reference = json.loads(_timecode_solve(files, level, None))
    return built


def _timecode_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    job = json.loads(files["job.json"])
    rate, drop = job["fps"], job["drop_frame"]
    if variant == "wrong_fps":
        if rate not in WRONG_FPS:
            return "{}"
        rate = WRONG_FPS[rate]
    if variant == "dropframe_as_nondrop":
        if not drop:
            return "{}"
        drop = False
    fps = NOMINAL[rate]
    inclusive = (job["out_point"] == "inclusive") != (variant == "inclusive_out_point")  # the other convention
    clips = {r["clip"]: _tc_frame(r["source_out"], fps, drop) - _tc_frame(r["source_in"], fps, drop) + inclusive
             for r in rows(files["edl.csv"])}
    total = sum(clips.values())
    if variant == "total_from_seconds_rounding":
        real = Fraction(30000, 1001) if drop else fps
        total = round(sum(round(f / real) for f in clips.values()) * real)
    label_drop = drop
    if variant == "dropframe_total_as_nondrop":
        if not drop:
            return "{}"
        label_drop = False
    return json.dumps({"clips": clips, "total_frames": total, "total_timecode": _tc_label(total, fps, label_drop)})


def _timecode_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    clips = out.get("clips")
    results = [isinstance(clips, dict) and set(clips) == set(reference["clips"])]
    if not results[0]:
        return results
    results += [type(clips[k]) is int and clips[k] == v for k, v in reference["clips"].items()]
    return results + [type(out.get("total_frames")) is int and out["total_frames"] == reference["total_frames"],
                      out.get("total_timecode") == reference["total_timecode"]]


# D3 catalogue: normalise a media library, dedupe it and build a capped playlist.
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
KINDS = ("film", "episode", "trailer", "music", "short")


def _iso_duration(seconds: int) -> str:
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    text = "".join(f"{v}{u}" for v, u in ((h, "H"), (m, "M"), (s, "S")) if v)
    return "PT" + (text or "0S")


def _date_text(day: date, rng: random.Random) -> str:
    style = rng.randrange(4)
    if style == 0:
        return day.isoformat()
    if style == 1:
        return f"{day.day:02d}.{day.month:02d}.{day.year}"
    if style == 2:
        return f"{day.year}/{day.month:02d}/{day.day:02d}"
    return f"{MONTHS[day.month - 1]} {day.day}, {day.year}"


def _parse_date(text: str) -> date:
    text = text.strip()
    if m := re.fullmatch(r"(\d{4})[-/](\d{2})[-/](\d{2})", text):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", text):
        return date(int(m[3]), int(m[2]), int(m[1]))
    m = re.fullmatch(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", text)
    return date(int(m[3]), MONTHS.index(m[1]) + 1, int(m[2]))


def _duration_text(seconds: int, style: int) -> str:
    if style == 0:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    if style == 1:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    return str(seconds)


def _parse_duration(text: str, misparse: bool = False) -> int:
    parts = [int(p) for p in text.strip().split(":")]
    if len(parts) == 2 and misparse:  # MM:SS read as H:MM
        return parts[0] * 3600 + parts[1] * 60
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def _messy_title(rng: random.Random) -> str:
    words = [rng.choice(WORDS) for _ in range(rng.randint(2, 5))]
    words[0] = words[0].capitalize()
    inner = "".join(w + " " * rng.choice((1, 1, 2, 3)) for w in words[:-1])
    return " " * rng.randint(0, 2) + inner + words[-1] + " " * rng.randint(0, 2)


def _catalogue_build(level: int, rng: random.Random) -> Built:
    count = {1: 6, 2: 8, 3: 10}[level]
    ids = [f"M{n:04d}" for n in rng.sample(range(1, 10000), count)]
    pool = [date(2023, 1, 1) + timedelta(days=rng.randint(0, 900)) for _ in range(4 if level == 3 else 20)]
    styles = [1, 0, 2] + [rng.randrange(3) for _ in range(count - 3)]  # every duration form at least once
    rng.shuffle(styles)
    records = []
    for item, style in zip(ids, styles):
        seconds = rng.randint(3700, 9000) if style == 0 else rng.randint(30, 3500)
        records.append([item, _messy_title(rng), _duration_text(seconds, style), rng.choice(KINDS),
                        rng.choice(pool)])
    rng.shuffle(records)
    unique = list(range(count))
    if level == 3:  # two short items share the newest date, listed against id order, so the tie-break shows
        a, b = sorted(rng.sample([k for k in unique if records[k][2].count(":") < 2], 2))
        if records[a][0] < records[b][0]:
            records[a][0], records[b][0] = records[b][0], records[a][0]
        records[a][4] = records[b][4] = max(pool) + timedelta(days=rng.randint(1, 30))
        unique = [k for k in unique if k not in (a, b)]
    if level >= 2:  # duplicate ids with a later, revised row; the first group lists the older row first
        for n, k in enumerate(sorted(rng.sample(unique, 2), reverse=True)):
            orig = records[k]
            later = orig[4] + timedelta(days=rng.randint(1, 60))
            seconds = _parse_duration(orig[2]) + rng.randint(5, 300)
            dup = [orig[0], _messy_title(rng), _duration_text(seconds, rng.randrange(3)), orig[3], later]
            records.insert(k + 1 if n == 0 else k, dup)
    rows_out = [r[:4] + [_date_text(r[4], rng)] for r in records]
    files = {"library.csv": write_csv(["id", "title", "duration", "type", "added"], rows_out)}
    lines = [
        "library.csv lists media items (id, title, duration, type, added). Normalise every row:",
        "- title: remove leading and trailing whitespace and replace every run of internal whitespace with one "
        "space; keep the letter case.",
        "- duration: given as H:MM:SS (hours, minutes, seconds), MM:SS (minutes, seconds) or a plain number of "
        "seconds. Write it as an ISO 8601 duration PT#H#M#S, leaving out every zero component and using no leading "
        'zeros (3725 seconds is "PT1H2M5S", 300 seconds is "PT5M").',
        "- added: given as YYYY-MM-DD, DD.MM.YYYY, YYYY/MM/DD or 'Mon D, YYYY' (English three-letter month). "
        "Write it as YYYY-MM-DD.",
        "- type: copy unchanged.",
    ]
    if level >= 2:
        lines.append("Some ids appear more than once. For each id keep only the row with the latest added date "
                     "(rows sharing an id never share a date).")
    out = '{"items": [{"id": ..., "title": ..., "duration": ..., "type": ..., "added": ...}, ...]}'
    if level == 3:
        cap = sum(_parse_duration(r[2]) for r in rows_out) * rng.randint(35, 55) // 100
        files["job.json"] = json.dumps({"cap_seconds": cap}, indent=2) + "\n"
        lines.append("Then build a playlist from the deduplicated, normalised items, using cap_seconds from "
                     "job.json: order the items by added date, newest first, breaking ties by id ascending; walk "
                     "that order and add each item whose duration still keeps the running total at or under "
                     "cap_seconds, skipping any item that does not fit and continuing with the next.")
        out = out[:-1] + ', "playlist": {"ids": [<ids in playlist order>], "total_duration": "<PT#H#M#S>"}}'
    lines.append(f"Write catalogue.json as {out}, with items sorted by id ascending and every item value a string.")
    built = Built(files=files, statement="\n".join(lines), output="catalogue.json", reference=None)
    built.reference = json.loads(_catalogue_solve(files, level, None))
    return built


def _catalogue_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    if variant in ("cap_exceeded", "tie_break_ignored") and level < 3 or variant == "dedupe_keep_first" and level < 2:
        return "{}"
    parsed = [dict(r, seconds=_parse_duration(r["duration"], variant == "duration_seconds_misparse"),
                   day=_parse_date(r["added"]), pos=n) for n, r in enumerate(rows(files["library.csv"]))]
    kept: dict[str, dict] = {}
    for r in parsed:
        if r["id"] not in kept or (variant != "dedupe_keep_first" and r["day"] > kept[r["id"]]["day"]):
            kept[r["id"]] = r
    items = [kept[k] for k in sorted(kept)]
    out: dict = {"items": [{"id": r["id"], "title": " ".join(r["title"].split()),
                            "duration": _iso_duration(r["seconds"]), "type": r["type"],
                            "added": r["day"].isoformat()} for r in items]}
    if level == 3:
        cap = json.loads(files["job.json"])["cap_seconds"]
        tie = (lambda r: r["pos"]) if variant == "tie_break_ignored" else (lambda r: r["id"])
        ranked = sorted(items, key=lambda r: (-r["day"].toordinal(), tie(r)))
        chosen, total = [], 0
        for r in ranked:
            if total + r["seconds"] <= cap:
                chosen.append(r["id"])
                total += r["seconds"]
            elif variant == "cap_exceeded":  # take the item that crosses the cap, then stop
                chosen.append(r["id"])
                total += r["seconds"]
                break
        out["playlist"] = {"ids": chosen, "total_duration": _iso_duration(total)}
    return json.dumps(out)


def _catalogue_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    items = out.get("items")
    results = [isinstance(items, list) and len(items) == len(reference["items"]),
               set(out) == set(reference)]
    if not all(results):
        return results
    results += [item == ref for item, ref in zip(items, reference["items"])]
    if "playlist" in reference:
        results.append(out["playlist"] == reference["playlist"])
    return results


# D4 assets: centred crop rectangles and output sizes for platform targets.
TARGETS = (("square", 1, 1), ("portrait", 4, 5), ("story", 9, 16), ("landscape", 16, 9), ("classic", 4, 3),
           ("cinema", 21, 9), ("tall", 2, 3))
OUT_HEIGHTS = (720, 1080, 1280, 1350, 1440, 1920)
ASSET_HEADER = ["target", "crop_x", "crop_y", "crop_w", "crop_h", "out_w", "out_h"]


def _assets_build(level: int, rng: random.Random) -> Built:
    width, height = rng.randint(1800, 6200), rng.randint(1200, 4200)
    chosen = rng.sample(TARGETS, {1: 1, 2: rng.randint(3, 4), 3: rng.randint(3, 4)}[level])
    job: dict = {"source": {"width": width, "height": height},
                 "targets": [{"name": n, "aspect": f"{aw}:{ah}", "out_height": rng.choice(OUT_HEIGHTS)}
                             for n, aw, ah in chosen]}
    if level == 3:
        job["safe_margin_pct"] = rng.randint(3, 9)
    files = {"job.json": json.dumps(job, indent=2) + "\n"}
    area, w_name, h_name = "source", "W", "H"
    lines = ["job.json gives the source image width W and height H in pixels, and platform targets, each with an "
             "aspect ratio aw:ah and an output height. For every target compute a crop rectangle and an output size."]
    if level == 3:
        area, w_name, h_name = "safe area", "SW", "SH"
        lines.append("The crop must lie inside the safe area: the source minus a margin of mx = floor(W x "
                     "safe_margin_pct / 100) pixels on the left and on the right, and my = floor(H x "
                     "safe_margin_pct / 100) on the top and on the bottom. Its size is SW = W - 2mx by "
                     "SH = H - 2my.")
    lines += [
        f"The crop is the largest rectangle with the target aspect that fits the {area}, centred in it. If "
        f"{w_name} x ah > {h_name} x aw, crop_h = {h_name} and crop_w = floor({h_name} x aw / ah); otherwise "
        f"crop_w = {w_name} and crop_h = floor({w_name} x ah / aw)."
        + (" Then round crop_w and crop_h down to the nearest even number." if level == 3 else ""),
        f"crop_x = {'mx + ' if level == 3 else ''}floor(({w_name} - crop_w) / 2) and "
        f"crop_y = {'my + ' if level == 3 else ''}floor(({h_name} - crop_h) / 2), measured from the source's "
        "top-left corner.",
        "out_h is the target's out_height and out_w = floor(out_h x aw / ah)"
        + (", rounded down to the nearest even number." if level == 3 else "."),
        "Write crops.csv with the header " + ",".join(ASSET_HEADER) + " and one row per target, in the order of "
        "job.json, every value but target an integer.",
    ]
    built = Built(files=files, statement="\n".join(lines), output="crops.csv", reference=None)
    built.reference = rows(_assets_solve(files, level, None))
    return built


def _assets_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    if variant in ("odd_dimensions", "ignore_safe_area") and level < 3:
        return "target\n"
    job = json.loads(files["job.json"])
    width, height = job["source"]["width"], job["source"]["height"]
    pct = 0 if variant == "ignore_safe_area" else job.get("safe_margin_pct", 0)
    mx, my = width * pct // 100, height * pct // 100
    sw, sh = width - 2 * mx, height - 2 * my

    def size(v: Fraction) -> int:
        if level < 3:
            return v.numerator // v.denominator
        return round(v) if variant == "odd_dimensions" else v.numerator // v.denominator // 2 * 2

    records = []
    for target in job["targets"]:
        aw, ah = (int(v) for v in target["aspect"].split(":"))
        if variant == "distort_aspect":
            x, y, cw, ch = 0, 0, width, height
        else:
            wide = sw * ah > sh * aw
            cw, ch = (Fraction(sh * aw, ah), Fraction(sh)) if wide else (Fraction(sw), Fraction(sw * ah, aw))
            cw, ch = size(cw), size(ch)
            x, y = (mx, my) if variant == "crop_not_centered" else (mx + (sw - cw) // 2, my + (sh - ch) // 2)
        out_h = target["out_height"]
        out_w = size(Fraction(out_h * aw, ah))
        records.append([target["name"], x, y, cw, ch, out_w, out_h])
    return write_csv(ASSET_HEADER, records)


def _assets_check(reference: list, text: str) -> list[bool]:
    got = rows(text)
    results = [len(got) == len(reference) and all(list(r) == ASSET_HEADER for r in got)]
    if not results[0]:
        return results
    for row, ref in zip(got, reference):
        results.append(row["target"] == ref["target"])
        results += [re.fullmatch(r"\d+", row[k] or "") is not None and int(row[k]) == int(ref[k])
                    for k in ASSET_HEADER[1:]]
    return results


FAMILIES = (
    Family(name="subtitles", domain=DOMAIN, build=_subtitles_build, check=_subtitles_check, solve=_subtitles_solve,
           traps={"wrong_sign": (1, 2, 3), "no_rate": (2, 3), "inverted_rate": (2, 3), "shift_first": (2, 3),
                  "no_sort": (3,), "no_gap": (3,), "no_wrap": (3,)}),
    Family(name="timecode", domain=DOMAIN, build=_timecode_build, check=_timecode_check, solve=_timecode_solve,
           traps={"wrong_fps": (1, 2), "inclusive_out_point": (1, 2, 3), "dropframe_as_nondrop": (3,),
                  "dropframe_total_as_nondrop": (3,), "total_from_seconds_rounding": (1, 2, 3)}),
    Family(name="catalogue", domain=DOMAIN, build=_catalogue_build, check=_catalogue_check, solve=_catalogue_solve,
           traps={"duration_seconds_misparse": (1, 2, 3), "dedupe_keep_first": (2, 3), "cap_exceeded": (3,),
                  "tie_break_ignored": (3,)}),
    Family(name="assets", domain=DOMAIN, build=_assets_build, check=_assets_check, solve=_assets_solve,
           traps={"distort_aspect": (1, 2, 3), "crop_not_centered": (1, 2, 3), "odd_dimensions": (3,),
                  "ignore_safe_area": (3,)}),
)
