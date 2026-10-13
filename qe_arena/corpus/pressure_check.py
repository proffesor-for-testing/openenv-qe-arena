"""Convert a pressure reading to the nominal's unit and check it against a tolerance. Original."""

import math
from fractions import Fraction

META = {
    "entry": "pressure_check",
    "origin": "Original.",
    "tags": ["units", "tolerance", "instrumentation"],
    "domain": "industrial-physical-systems",
}


def pressure_check(reading_milli, reading_unit, nominal_milli, nominal_unit, tolerance_bp):
    """Check a pressure reading against a nominal value that may be in another unit.

    Values are ints in thousandths of their unit (2500 "bar" is 2.5 bar). Units in pascals, exact:
    "Pa" 1, "kPa" 1000, "bar" 100000, "psi" 6894.757293168, "mmHg" 133.322387415.
    converted is the reading in thousandths of nominal_unit, rounded half up. deviation_bp is
    (reading - nominal) / nominal * 10000, exact, truncated toward zero. verdict is "ok" if
    abs(deviation_bp) <= tolerance_bp, else "low" or "high" by the sign of deviation_bp.
    Returns [converted, deviation_bp, verdict].
    Raises ValueError if nominal_milli < 1, reading_milli < 0 or tolerance_bp < 0 (checked
    first), and KeyError for an unknown unit.
    """
    pascals = {"Pa": Fraction(1), "kPa": Fraction(1000), "bar": Fraction(100000),
               "psi": Fraction(6894757293168, 10**9), "mmHg": Fraction(133322387415, 10**9)}
    if nominal_milli < 1 or reading_milli < 0 or tolerance_bp < 0:
        raise ValueError("invalid pressure check")
    reading = reading_milli * pascals[reading_unit]
    nominal = nominal_milli * pascals[nominal_unit]
    converted = math.floor(reading / pascals[nominal_unit] + Fraction(1, 2))
    deviation_bp = math.trunc((reading - nominal) * 10000 / nominal)
    if abs(deviation_bp) <= tolerance_bp:
        return [converted, deviation_bp, "ok"]
    return [converted, deviation_bp, "low" if deviation_bp < 0 else "high"]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [2500, "bar", 2500, "bar", 0],
        [2500, "bar", 250000, "kPa", 0],
        [36259, "psi", 2500, "bar", 0],
        [36259, "psi", 2500, "bar", 1],
        [36259, "psi", 2500, "bar", 10],
        [36260, "psi", 2500, "bar", 0],
        [760000, "mmHg", 101325, "kPa", 0],
        [760000, "mmHg", 101325, "kPa", 1],
        [1013250, "Pa", 760000, "mmHg", 5],
        [2525, "bar", 2500, "bar", 100],
        [2526, "bar", 2500, "bar", 100],
        [2475, "bar", 2500, "bar", 100],
        [2474, "bar", 2500, "bar", 100],
        [0, "kPa", 1, "kPa", 9999],
        [1, "bar", 1, "Pa", 0],
        [1, "Pa", 1, "bar", 0],
        [3, "Pa", 2, "kPa", 0],
        [2500, "bar", 0, "bar", 0],
        [-1, "bar", 2500, "bar", 0],
        [2500, "bar", 2500, "bar", -1],
        [2500, "atm", 2500, "bar", 0],
        [2500, "bar", 2500, "torr", 0],
    ]
    units = ["Pa", "kPa", "bar", "psi", "mmHg"]
    scale = {"Pa": 100000000, "kPa": 100000, "bar": 1000, "psi": 14504, "mmHg": 750062}
    while len(cases) < n:
        a, b = rng.choice(units), rng.choice(units)
        nominal = scale[b] * rng.randint(1, 40) // 10 + rng.randint(0, 9)
        drift = rng.choice([0, 1, -1, 5, -5, 50, -50, 300, -300, rng.randint(-2000, 2000)])
        reading = max(0, (nominal * (10000 + drift) // 10000) * scale[a] // scale[b])
        cases.append([reading, a, nominal, b, rng.choice([0, 1, 10, 50, 100, 250])])
    return cases[:n]
