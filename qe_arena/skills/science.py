"""natural-science families for qe_skill."""

from __future__ import annotations

import json
import math
import random
from statistics import fmean
from typing import Any

from . import Built, Family
from ._common import close, is_number, load_object, rows, write_csv

DOMAIN = "natural-science"

# S1 units: per-sample mean concentration from a QC'd measurement sheet.
TO_MG_PER_L = {"mg/L": 1.0, "ug/L": 0.001, "ng/mL": 0.001, "g/L": 1000.0, "mg/mL": 1000.0}
TARGETS = ("mg/L", "ug/L")
ANALYTES = ("nitrate", "phosphate", "sulfate", "ammonium", "chloride", "fluoride")
TOLERANCE = 0.0006  # the statement asks for 3 decimals


def _units_build(level: int, rng: random.Random) -> Built:
    target = rng.choice(TARGETS)
    analytes = rng.sample(ANALYTES, 2 if level == 3 else 1)
    analyte = analytes[0]
    samples = [f"S{n:02d}" for n in rng.sample(range(1, 60), {1: 4, 2: 5, 3: 6}[level])]
    header = ["sample_id", "analyte", "value", "unit", "qc"] + (["lod"] if level == 3 else [])
    records: list[list[Any]] = []
    for sample in samples:
        for name in analytes:
            true = rng.uniform(0.2, 40.0)
            count = rng.randint(3, 5)
            fails = rng.sample(range(count), 1)
            nd = rng.sample([i for i in range(count) if i not in fails], 1) if level == 3 and name == analyte else []
            for i in range(count):
                unit = target if level == 1 else rng.choice(list(TO_MG_PER_L))
                factor = TO_MG_PER_L[unit]
                if i in nd:
                    lod = true * rng.uniform(0.2, 0.6) / factor
                    records.append([sample, name, "ND", unit, "ok", f"{lod:.4g}"])
                    continue
                value = true * rng.gauss(1.0, 0.05) * (rng.choice((0.2, 4.0)) if i in fails else 1.0) / factor
                lod = true * 0.05 / factor
                records.append([sample, name, f"{value:.6g}", unit, "fail" if i in fails else "ok"]
                               + ([f"{lod:.4g}"] if level == 3 else []))
    rng.shuffle(records)
    files = {"measurements.csv": write_csv(header, records),
             "request.json": json.dumps({"analyte": analyte, "unit": target}, indent=2) + "\n"}
    rules = ["Use only rows whose qc is ok."]
    if level >= 2:
        rules.append("Values come in several units; convert every value to the requested unit before averaging "
                     "(1 g/L = 1000 mg/L = 1 mg/mL; 1 mg/L = 1000 ug/L; 1 ng/mL = 1 ug/L).")
    if level == 3:
        rules.append("The sheet holds more than one analyte; report only the requested one. A value of ND means not "
                     "detected: count it as half of that row's lod (lod is in the row's own unit).")
    statement = "\n".join([
        "A water lab sent measurements.csv and request.json. Report the mean concentration of the requested "
        "analyte for every sample, in the requested unit.",
        *rules,
        'Write result.json as {"unit": "<the requested unit>", "means": {"<sample_id>": <mean>, ...}} with one '
        "entry per sample, each mean rounded to 3 decimal places.",
    ])
    built = Built(files=files, statement=statement, output="result.json", reference=None)
    built.reference = json.loads(_units_solve(files, level, None))
    return built


def _units_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    request = json.loads(files["request.json"])
    target = request["unit"]
    values: dict[str, list[float]] = {}
    for row in rows(files["measurements.csv"]):
        if row["analyte"] != request["analyte"] and variant != "all_analytes":
            continue
        if row["qc"] != "ok" and variant != "keep_flagged":
            continue
        factor = 1.0 if variant == "no_conversion" else TO_MG_PER_L[row["unit"]] / TO_MG_PER_L[target]
        if row["value"] == "ND":
            if variant == "nd_dropped":
                continue
            raw = 0.0 if variant == "nd_as_zero" else float(row["lod"]) / 2
        else:
            raw = float(row["value"])
        values.setdefault(row["sample_id"], []).append(raw * factor)
    means = {sample: round(fmean(v), 3) for sample, v in sorted(values.items())}
    return json.dumps({"unit": target, "means": means}, indent=2) + "\n"


def _units_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    means = out.get("means")
    results = [out.get("unit") == reference["unit"],
               isinstance(means, dict) and set(means) == set(reference["means"])]
    if not all(results):
        return results
    tol = lambda ref: TOLERANCE + 1e-9 * abs(ref)  # noqa: E731
    return results + [close(means[s], ref, tol(ref)) for s, ref in reference["means"].items()]



# Shared numeric helpers for the families below.
def _fit(xs: list[float], ys: list[float], origin: bool = False) -> tuple[float, float, float]:
    """Ordinary least squares y = slope*x + intercept: (slope, intercept, 1 - SS_res/SS_tot)."""
    mx, my = fmean(xs), fmean(ys)
    if origin:
        slope, intercept = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs), 0.0
    else:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        intercept = my - slope * mx
    ss_res = sum((y - slope * x - intercept) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, 1 - ss_res / sum((y - my) ** 2 for y in ys)


def _sig(value: float, digits: int) -> float:
    return float(f"{value:.{digits - 1}e}")


def _ulp(reference: float, digits: int) -> float:
    """One unit in the last of ``digits`` significant figures of the reference."""
    return 10 ** (math.floor(math.log10(abs(reference))) - digits + 1) if reference else 10 ** -digits


def _cell(text: str | None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _relative(value: Any, reference: float, rel: float) -> bool:
    return close(value, reference, rel * abs(reference) + 1e-12)


# S2 calibration: a linear calibration curve and the unknowns read off it.
def _calibration_build(level: int, rng: random.Random) -> Built:
    top = rng.choice((5, 10, 20, 25, 50))
    count = rng.randint(6, 8)
    concs = [round(top * k / (count - 1), 3) for k in range(count)]
    slope = rng.uniform(0.6, 1.0) / top
    intercept = rng.uniform(0.015, 0.05)
    absorb = [slope * c + intercept + rng.gauss(0, 0.004) for c in concs]
    if level == 3:
        absorb[rng.randrange(count)] += rng.choice((1, -1)) * rng.uniform(0.07, 0.12)
    blank = rng.uniform(0.03, 0.08) if level >= 2 else 0.0
    samples = [[f"U{n:02d}", f"{slope * rng.uniform(0.1, 0.9) * top + intercept + blank + rng.gauss(0, 0.004):.4f}"]
               for n in rng.sample(range(1, 40), rng.randint(3, 5))]
    if level >= 2:
        samples += [["BLANK", f"{blank + rng.gauss(0, 0.002):.4f}"] for _ in range(3)]
    rng.shuffle(samples)
    files = {"standards.csv": write_csv(["concentration_mg_L", "absorbance"],
                                        [[c, f"{a:.4f}"] for c, a in zip(concs, absorb)]),
             "samples.csv": write_csv(["sample_id", "absorbance"], samples)}
    rules = ["Fit absorbance = slope * concentration + intercept to the standards by ordinary least squares, then "
             "convert each sample's absorbance to a concentration with that line."]
    if level >= 2:
        rules.append("The standards are already blank-corrected; the sample readings are not. The rows with "
                     "sample_id BLANK are reagent blank readings: subtract their mean from every other sample reading "
                     "before converting it, and do not report the blanks.")
    if level == 3:
        files["job.json"] = json.dumps({"outlier_threshold": 0.03}, indent=2) + "\n"
        rules.append("After the first fit, find the one standard with the largest absolute residual (its absorbance "
                     "minus the fitted absorbance). If that residual's magnitude exceeds outlier_threshold in job.json, "
                     "drop that standard and fit again; report the second fit. Drop at most one standard.")
    statement = "\n".join([
        "A spectrophotometry run produced standards.csv (concentration_mg_L, absorbance) and samples.csv "
        "(sample_id, absorbance).",
        *rules,
        'Write calibration.json as {"slope": <number>, "intercept": <number>, "r_squared": <number>, '
        '"concentrations": {"<sample_id>": <mg/L>, ...}}. Give slope and intercept to 6 significant figures, '
        "r_squared (1 - SS_res/SS_tot over the standards in the reported fit) to 6 decimal places, and each "
        "concentration in mg/L to 3 decimal places.",
    ])
    built = Built(files=files, statement=statement, output="calibration.json", reference=None)
    built.reference = json.loads(_calibration_solve(files, level, None))
    return built


def _calibration_fit(xs: list[float], ys: list[float], variant: str | None) -> tuple[float, float, float]:
    if variant == "inverted_axes":  # concentration regressed on absorbance, then the line turned around
        a, b, r2 = _fit(ys, xs)
        return 1 / a, -b / a, r2
    return _fit(xs, ys, origin=variant == "through_origin")


def _calibration_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    std = [(float(r["concentration_mg_L"]), float(r["absorbance"])) for r in rows(files["standards.csv"])]
    slope, intercept, r2 = _calibration_fit([c for c, _ in std], [a for _, a in std], variant)
    if level == 3 and variant != "outlier_kept":
        threshold = json.loads(files["job.json"])["outlier_threshold"]
        worst = max(range(len(std)), key=lambda i: abs(std[i][1] - slope * std[i][0] - intercept))
        if abs(std[worst][1] - slope * std[worst][0] - intercept) > threshold:
            kept = std[:worst] + std[worst + 1:]
            slope, intercept, r2 = _calibration_fit([c for c, _ in kept], [a for _, a in kept], variant)
    readings = [(r["sample_id"], float(r["absorbance"])) for r in rows(files["samples.csv"])]
    blanks = [a for sid, a in readings if sid == "BLANK"]
    blank = fmean(blanks) if blanks and variant != "no_blank" else 0.0
    concentrations = {sid: round((a - blank - intercept) / slope, 3) for sid, a in sorted(readings) if sid != "BLANK"}
    return json.dumps({"slope": _sig(slope, 6), "intercept": _sig(intercept, 6), "r_squared": round(r2, 6),
                       "concentrations": concentrations}, indent=2) + "\n"


def _calibration_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    concentrations = out.get("concentrations")
    results = [_relative(out.get("slope"), reference["slope"], 1e-5),
               _relative(out.get("intercept"), reference["intercept"], 1e-5),
               close(out.get("r_squared"), reference["r_squared"], 6e-7),
               isinstance(concentrations, dict) and set(concentrations) == set(reference["concentrations"])]
    if not all(results):
        return results
    return results + [close(concentrations[s], ref, 0.0006 + 1e-9 * abs(ref))
                      for s, ref in reference["concentrations"].items()]


# S3 decay: half-life from a count-rate series, with background and dead-time corrections.
def _decay_build(level: int, rng: random.Random) -> Built:
    half = rng.uniform(3.0, 40.0)
    count = rng.randint(10, 14)
    step = max(0.5, round(half * rng.uniform(3.0, 4.0) / (count - 1) * 2) / 2)
    times = [k * step for k in range(count)]
    start = rng.uniform(2000.0, 8000.0)
    signal = [start * math.exp(-math.log(2) / half * t) * rng.gauss(1.0, 0.004) for t in times]
    background = round(signal[-1] * rng.uniform(0.3, 0.8), 2) if level >= 2 else 0.0
    dead = float(f"{rng.uniform(0.1, 0.2) / (start + background):.3g}") if level == 3 else 0.0
    observed = [(s + background) / (1 + (s + background) * dead) for s in signal]
    files = {"counts.csv": write_csv(["t_min", "rate_cps"], [[f"{t:g}", f"{m:.2f}"] for t, m in zip(times, observed)])}
    rules = []
    if level == 2:
        files["job.json"] = json.dumps({"background_cps": background}, indent=2) + "\n"
        rules.append("Every reading includes the detector background, background_cps in job.json: subtract it from "
                     "each rate first.")
    if level == 3:
        files["job.json"] = json.dumps({"background_cps": background, "dead_time_s": dead}, indent=2) + "\n"
        rules.append("The readings are not corrected for detector dead time or background (job.json). First correct "
                     "each observed rate m for dead time (non-paralysable): m / (1 - m * dead_time_s). Then subtract "
                     "background_cps.")
    statement = "\n".join([
        "counts.csv holds the observed count rate (rate_cps, counts per second) of a radioactive sample at times "
        "t_min (minutes).",
        *rules,
        "Fit a straight line to the natural log of the net rate against time by ordinary least squares, every point "
        "weighted equally; the decay constant is minus the slope, and the half-life is ln(2) divided by it.",
        'Write decay.json as {"half_life_min": <minutes>, "decay_constant_per_min": <per minute>}, each to 5 '
        "significant figures. Values within 1% of the expected ones are accepted.",
    ])
    built = Built(files=files, statement=statement, output="decay.json", reference=None)
    built.reference = json.loads(_decay_solve(files, level, None))
    return built


def _decay_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    job = json.loads(files.get("job.json", "{}"))
    times, net = [], []
    for row in rows(files["counts.csv"]):
        m = float(row["rate_cps"])
        if level == 3 and variant != "no_dead_time":
            m = m / (1 - m * job["dead_time_s"])
        if level >= 2 and variant != "no_background":
            m -= job["background_cps"]
        times.append(float(row["t_min"]))
        net.append(m)
    if variant == "linear_fit_raw":  # a straight line through the rates, read off where it halves
        slope, intercept, _ = _fit(times, net)
        half = (intercept / 2) / -slope
        rate = math.log(2) / half
    else:
        logs = [math.log10(v) if variant == "log10_instead_of_ln" else math.log(v) for v in net]
        rate = -_fit(times, logs)[0]
        half = math.log(2) / rate
    return json.dumps({"half_life_min": _sig(half, 5), "decay_constant_per_min": _sig(rate, 5)}, indent=2) + "\n"


def _decay_check(reference: dict, text: str) -> list[bool]:
    out = load_object(text)
    return [_relative(out.get(key), ref, 0.01) for key, ref in reference.items()]


# S4 dilution: a pipetting plan by C1V1 = C2V2, serial chains, and a minimum transfer volume.
VOLUMES_UL = (200, 500, 1000, 1500, 2000, 5000)
FACTORS = (2, 2.5, 3, 4, 6, 7, 8, 12, 15)
BIG_FACTORS = (450, 750, 1600, 2500, 6000)  # never an exact min_transfer_uL boundary


def _dilution_build(level: int, rng: random.Random) -> Built:
    if level == 1:
        stocks = {f"ST{n}": rng.choice((10, 25, 50, 100, 250, 500)) for n in rng.sample(range(1, 10), 2)}
        targets = []
        for n in rng.sample(range(1, 50), rng.randint(4, 6)):
            stock = rng.choice(sorted(stocks))
            targets.append([f"T{n:02d}", stock, f"{_sig(stocks[stock] * rng.uniform(0.01, 0.3), 3):g}",
                            rng.choice(VOLUMES_UL)])
        files = {"stocks.csv": write_csv(["stock_id", "concentration_mM"], sorted(stocks.items())),
                 "targets.csv": write_csv(["target_id", "stock_id", "target_mM", "final_uL"], targets)}
        statement = "\n".join([
            "stocks.csv lists stock solutions (stock_id, concentration_mM) and targets.csv lists solutions to prepare "
            "(target_id, stock_id, target_mM, final_uL). Each target is made by diluting its stock with diluent to "
            "the final volume (C1 * V1 = C2 * V2).",
            "Write plan.csv with the header target_id,stock_uL,diluent_uL and one row per target. Round stock_uL to "
            "1 decimal place; diluent_uL is final_uL minus the rounded stock_uL.",
        ])
        built = Built(files=files, statement=statement, output="plan.csv", reference=None)
        built.reference = _dilution_parse(_dilution_solve(files, 1, None))
        return built
    steps = []
    for n in range(rng.randint(4, 6) if level == 3 else rng.randint(5, 7)):
        steps.append({"tube": f"D{n + 1}", "dilution_factor": rng.choice(FACTORS), "final_uL": rng.choice(VOLUMES_UL[:4])})
    job: dict[str, Any] = {"stock_mM": float(f"{rng.uniform(5, 500):.4g}"), "steps": steps}
    rules = ["Rows follow the order of steps. Compute every concentration from the unrounded concentration of its "
             "source; round only the values you write."]
    if level == 3:
        steps[rng.randrange(len(steps))]["dilution_factor"] = rng.choice(BIG_FACTORS)
        job.update(min_transfer_uL=5.0, intermediate_factor=10, intermediate_uL=500)
        rules.append("No transfer may be below min_transfer_uL. When a tube's transfer would be below it, first "
                     "make an intermediate tube named <tube>-I1 from that tube's source, diluted intermediate_factor-"
                     "fold to intermediate_uL, and make the tube from the intermediate with the remaining factor "
                     "(dilution_factor / intermediate_factor). If the transfer is still below the minimum, insert "
                     "<tube>-I2 made from <tube>-I1 the same way, dividing the remaining factor by "
                     "intermediate_factor again, and so on. Intermediates get their own rows, just before their "
                     "tube; the next tube in steps is still made from the previous tube, never from an intermediate.")
    files = {"serial.json": json.dumps(job, indent=2) + "\n"}
    statement = "\n".join([
        "serial.json describes a serial dilution: a stock at stock_mM and a list of steps. The first tube is made "
        "from the stock and every later tube from the tube before it: transfer final_uL / dilution_factor of the "
        "source and add diluent to final_uL.",
        *rules,
        "Write plan.csv with the header tube,source,transfer_uL,diluent_uL,concentration_mM and one row per tube "
        "(source is stock or a tube name). Round transfer_uL to 1 decimal place; diluent_uL is final_uL minus the "
        "rounded transfer_uL; give concentration_mM to 4 significant figures.",
    ])
    built = Built(files=files, statement=statement, output="plan.csv", reference=None)
    built.reference = _dilution_parse(_dilution_solve(files, level, None))
    return built


def _dilution_solve(files: dict[str, str], level: int, variant: str | None) -> str:
    def transfer(final: float, factor: float) -> float:
        return round(final * factor if variant == "inverted_c1v1" else final / factor, 1)

    if level == 1:
        stocks = {r["stock_id"]: float(r["concentration_mM"]) for r in rows(files["stocks.csv"])}
        records = []
        for r in rows(files["targets.csv"]):
            final = float(r["final_uL"])
            stock = transfer(final, stocks[r["stock_id"]] / float(r["target_mM"]))
            records.append([r["target_id"], f"{stock:.1f}", f"{final - stock:.1f}"])
        return write_csv(["target_id", "stock_uL", "diluent_uL"], records)
    job = json.loads(files["serial.json"])
    records: list[list[Any]] = []
    carry = job["stock_mM"]
    source = "stock"

    def make(tube: str, src: str, conc: float, factor: float, final: float) -> float:
        conc = conc / factor
        if variant == "round_each_step":
            conc = _sig(conc, 4)
        amount = transfer(final, factor)
        records.append([tube, src, f"{amount:.1f}", f"{final - amount:.1f}", f"{_sig(conc, 4):g}"])
        return conc

    for step in job["steps"]:
        factor, final, conc, src = step["dilution_factor"], step["final_uL"], carry, source
        k = 0
        while level == 3 and variant != "ignore_min_volume" and final / factor < job["min_transfer_uL"]:
            k += 1
            conc = make(f"{step['tube']}-I{k}", src, conc, job["intermediate_factor"], job["intermediate_uL"])
            src, factor = f"{step['tube']}-I{k}", factor / job["intermediate_factor"]
        carry, source = make(step["tube"], src, conc, factor, final), step["tube"]
    return write_csv(["tube", "source", "transfer_uL", "diluent_uL", "concentration_mM"], records)


def _dilution_parse(text: str) -> list[dict[str, Any]]:
    return [{key: (value if key in ("target_id", "tube", "source") else _cell(value)) for key, value in row.items()}
            for row in rows(text)]


def _dilution_check(reference: list[dict[str, Any]], text: str) -> list[bool]:
    out = _dilution_parse(text)
    ids = ("target_id",) if "target_id" in reference[0] else ("tube", "source")
    key = lambda row: tuple(row.get(k) for k in ids)  # noqa: E731
    if ids == ("target_id",):  # targets may come in any order; serial rows must follow the steps
        out = sorted(out, key=lambda row: str(row.get("target_id")))
        reference = sorted(reference, key=lambda row: row["target_id"])
    results = [len(out) == len(reference) and [key(r) for r in out] == [key(r) for r in reference]]
    if not all(results):
        return results
    for got, ref in zip(out, reference):
        for name, value in ref.items():
            if name in ids:
                continue
            tol = 0.6 * _ulp(value, 4) if name == "concentration_mM" else 0.06
            results.append(is_number(got.get(name)) and close(got[name], value, tol))
    return results


FAMILIES = (
    Family(name="units", domain=DOMAIN, build=_units_build, check=_units_check, solve=_units_solve,
           traps={"keep_flagged": (1, 2, 3), "no_conversion": (2, 3), "nd_as_zero": (3,), "nd_dropped": (3,),
                  "all_analytes": (3,)}),
    Family(name="calibration", domain=DOMAIN, build=_calibration_build, check=_calibration_check,
           solve=_calibration_solve,
           traps={"no_blank": (2, 3), "through_origin": (1, 2, 3), "outlier_kept": (3,), "inverted_axes": (1, 2, 3)}),
    Family(name="decay", domain=DOMAIN, build=_decay_build, check=_decay_check, solve=_decay_solve,
           traps={"linear_fit_raw": (1, 2, 3), "no_background": (2, 3), "no_dead_time": (3,),
                  "log10_instead_of_ln": (1, 2, 3)}),
    Family(name="dilution", domain=DOMAIN, build=_dilution_build, check=_dilution_check, solve=_dilution_solve,
           traps={"inverted_c1v1": (1, 2, 3), "round_each_step": (2, 3), "ignore_min_volume": (3,)}),
)
