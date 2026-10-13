"""Choose the next rung of an adaptive-streaming bitrate ladder from bandwidth and buffer. Original."""

META = {
    "entry": "select_bitrate_rung",
    "origin": "Original.",
    "tags": ["streaming", "abr", "video"],
    "domain": "media-content-production",
}


def select_bitrate_rung(ladder, bandwidth_kbps, buffer_ms, current):
    """Pick the ladder rung an adaptive video player should play next; return [index, reason].

    ladder is a list of [height, kbps] rungs with strictly ascending kbps; current is the
    index of the rung playing now. The sustainable rung is the highest index whose
    kbps * 100 <= bandwidth_kbps * 80 (an 80% safety margin), or 0 when no rung fits.
    The rules, first match wins:
    - buffer_ms < 4000: [0, "panic"];
    - sustainable > current: [current + 1, "up"] (one rung at a time) when buffer_ms >= 12000,
      else [current, "hold"];
    - sustainable < current: [sustainable, "down"] (down-switches are immediate);
    - otherwise [current, "stay"].
    Raises ValueError for an empty ladder, kbps that do not strictly ascend, current outside
    0..len(ladder) - 1, or a negative bandwidth_kbps or buffer_ms.
    """
    if not ladder or not 0 <= current < len(ladder):
        raise ValueError("current must index a non-empty ladder")
    if bandwidth_kbps < 0 or buffer_ms < 0:
        raise ValueError("bandwidth and buffer must be >= 0")
    for lower, upper in zip(ladder, ladder[1:]):
        if upper[1] <= lower[1]:
            raise ValueError("ladder kbps must strictly ascend")
    if buffer_ms < 4000:
        return [0, "panic"]
    sustainable = 0
    for index, (height, kbps) in enumerate(ladder):
        if kbps * 100 <= bandwidth_kbps * 80:
            sustainable = index
    if sustainable > current:
        return [current + 1, "up"] if buffer_ms >= 12000 else [current, "hold"]
    if sustainable < current:
        return [sustainable, "down"]
    return [current, "stay"]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    ladder = [[240, 400], [360, 800], [480, 1400], [720, 2800], [1080, 5000], [1440, 8000], [2160, 16000]]
    cases = [
        [ladder, 6250, 20000, 3],
        [ladder, 6249, 20000, 3],
        [ladder, 6250, 12000, 3],
        [ladder, 6250, 11999, 3],
        [ladder, 6250, 4000, 4],
        [ladder, 6250, 3999, 4],
        [ladder, 0, 30000, 2],
        [ladder, 5000, 0, 2],
        [ladder, 499, 30000, 0],
        [ladder, 1000, 30000, 1],
        [ladder, 50000, 30000, 6],
        [ladder, 3500, 9000, 6],
        [[[720, 3000]], 1000, 20000, 0],
        [ladder, 5000, 20000, 7],
        [ladder, 5000, 20000, -1],
        [[], 5000, 20000, 0],
        [[[360, 800], [480, 800]], 5000, 20000, 0],
        [ladder, -1, 20000, 0],
        [ladder, 5000, -1, 0],
    ]
    while len(cases) < n:
        cases.append([ladder, rng.randint(0, 25000), rng.randint(0, 30000), rng.randint(0, 6)])
    return cases[:n]
