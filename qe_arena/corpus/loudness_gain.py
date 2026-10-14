"""Compute a loudness-normalisation gain in centi-dB, capped by boost and true-peak limits. Original."""

META = {
    "entry": "loudness_gain",
    "origin": "Original.",
    "tags": ["audio", "loudness", "normalization"],
    "domain": "media-content-production",
}


def loudness_gain(measured_cdb, peak_cdb, target_cdb, ceiling_cdb, max_boost_cdb):
    """Loudness-normalisation gain in integer centi-dB (1 dB = 100); return [gain, limited_by].

    measured_cdb is integrated loudness (LUFS * 100), peak_cdb the true peak (dBTP * 100).
    Below -7000 measured is silence: [0, "silence"]. Otherwise gain = target_cdb -
    measured_cdb ("target"); if gain > max_boost_cdb it becomes max_boost_cdb ("max_boost");
    then if peak_cdb + gain > ceiling_cdb it becomes ceiling_cdb - peak_cdb ("peak"), which
    may be a cut. Raises ValueError first if max_boost_cdb < 0 or ceiling_cdb > 0.
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
    return [gain, limited_by]


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
        ceiling = rng.choice([-100, -100, -200, -150, 0, 0, -100, 100])  # 100 is an invalid ceiling
        boost = rng.choice([0, 300, 600, 1200, rng.randint(0, 2000), rng.randint(0, 2000), rng.randint(0, 2000), -300])
        cases.append([measured, peak, rng.choice(targets), ceiling, boost])
    return cases[:n]
