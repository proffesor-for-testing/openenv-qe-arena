"""Choose the next rung of an adaptive-streaming bitrate ladder from bandwidth and buffer. Original."""

META = {
    "entry": "select_bitrate_rung",
    "origin": "Original.",
    "tags": ["streaming", "abr", "video"],
    "domain": "media-content-production",
}


def select_bitrate_rung(bandwidth_kbps, buffer_ms, current):
    """Pick the next rung of the ladder 400, 1200, 3000, 6000 kbps; return [index, reason].

    current is the rung (0-3) playing now. The sustainable rung is the highest whose
    kbps * 100 <= bandwidth_kbps * 80. First match wins: buffer_ms < 4000 gives [0, "panic"];
    no sustainable rung gives [0, "starved"]; sustainable > current gives [current + 1, "up"]
    if buffer_ms >= 12000, else [current, "hold"]; sustainable < current gives
    [sustainable, "down"]; otherwise [current, "stay"]. Raises ValueError if current is not
    0-3 or bandwidth_kbps or buffer_ms is negative.
    """
    ladder = [400, 1200, 3000, 6000]
    if not 0 <= current < len(ladder) or bandwidth_kbps < 0 or buffer_ms < 0:
        raise ValueError("bad player state")
    if buffer_ms < 4000:
        return [0, "panic"]
    sustainable = None
    for index, kbps in enumerate(ladder):
        if kbps * 100 <= bandwidth_kbps * 80:
            sustainable = index
    if sustainable is None:
        return [0, "starved"]
    if sustainable > current:
        return [current + 1, "up"] if buffer_ms >= 12000 else [current, "hold"]
    if sustainable < current:
        return [sustainable, "down"]
    return [current, "stay"]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [500, 20000, 1],
        [499, 20000, 1],
        [1500, 20000, 0],
        [1499, 20000, 0],
        [3750, 20000, 1],
        [3749, 20000, 1],
        [7500, 20000, 2],
        [7499, 20000, 2],
        [3750, 12000, 1],
        [3750, 11999, 1],
        [3750, 4000, 3],
        [3750, 3999, 3],
        [0, 30000, 2],
        [5000, 0, 2],
        [50000, 30000, 3],
        [5000, 20000, 4],
        [5000, 20000, -1],
        [-1, 20000, 0],
        [5000, -1, 0],
    ]
    while len(cases) < n:
        current = rng.randint(0, 3) if rng.random() < 0.95 else rng.choice([-1, 4])
        cases.append([rng.randint(0, 9000), rng.randint(0, 30000), current])
    return cases[:n]
