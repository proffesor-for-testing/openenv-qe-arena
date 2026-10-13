"""Compute a loudness-normalisation gain in centi-dB, capped by boost and true-peak limits. Original."""

META = {
    "entry": "loudness_gain",
    "origin": "Original.",
    "tags": ["audio", "loudness", "normalization"],
    "domain": "media-content-production",
}


def loudness_gain(measured_cdb, peak_cdb, target_cdb, ceiling_cdb, max_boost_cdb):
    """Return the playback gain that normalises a track to a loudness target, in centi-dB.

    All values are integers in centi-dB (1 dB = 100): measured_cdb is the track's integrated
    loudness (LUFS * 100, so -2340 is -23.4 LUFS), peak_cdb its true peak (dBTP * 100),
    target_cdb the target loudness, ceiling_cdb the highest true peak allowed after gain,
    max_boost_cdb the largest positive gain allowed. Steps, in order:
    1. measured_cdb below -7000 (quieter than -70 LUFS) is treated as silence: gain 0,
       limited_by "silence";
    2. gain = target_cdb - measured_cdb, limited_by "target";
    3. if gain > max_boost_cdb: gain = max_boost_cdb, limited_by "max_boost";
    4. if peak_cdb + gain > ceiling_cdb: gain = ceiling_cdb - peak_cdb, limited_by "peak"
       (this may turn a boost into a cut).
    Returns {"gain_cdb": gain, "loudness_cdb": measured_cdb + gain, "peak_cdb": peak_cdb + gain,
    "limited_by": the label}. Raises ValueError if max_boost_cdb is negative or ceiling_cdb
    is above 0 (checked before anything else).
    """
    if max_boost_cdb < 0 or ceiling_cdb > 0:
        raise ValueError("need max_boost_cdb >= 0 and ceiling_cdb <= 0")
    if measured_cdb < -7000:
        gain, limited_by = 0, "silence"
    else:
        gain, limited_by = target_cdb - measured_cdb, "target"
        if gain > max_boost_cdb:
            gain, limited_by = max_boost_cdb, "max_boost"
        if peak_cdb + gain > ceiling_cdb:
            gain, limited_by = ceiling_cdb - peak_cdb, "peak"
    return {"gain_cdb": gain, "loudness_cdb": measured_cdb + gain, "peak_cdb": peak_cdb + gain,
            "limited_by": limited_by}


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [-2300, -600, -1400, -100, 1200],
        [-1000, -50, -1400, -100, 1200],
        [-2600, -300, -1400, -100, 1200],
        [-2600, -1300, -1400, -100, 1200],
        [-2600, -1299, -1400, -100, 1200],
        [-2400, -1100, -1400, -100, 1000],
        [-2400, -1101, -1400, -100, 999],
        [-7000, -4000, -2300, -100, 600],
        [-7001, -4000, -2300, -100, 600],
        [-900, 150, -1400, 0, 0],
        [-1400, -100, -1400, -100, 0],
        [-2000, -500, -1600, 0, 1200],
        [-2000, -500, -1600, -100, -1],
        [-2000, -500, -1600, 1, 1200],
        [-9000, 200, -1400, 5, 100],
    ]
    targets = [-1400, -1600, -2300, -2400]
    while len(cases) < n:
        measured = rng.choice([rng.randint(-3500, -400), rng.randint(-8000, -6000)])
        peak = rng.randint(measured + 300, max(measured + 400, 300))
        cases.append([measured, peak, rng.choice(targets), rng.choice([-100, -100, -200, 0]),
                      rng.choice([0, 300, 600, 1200, rng.randint(0, 2000)])])
    return cases[:n]
