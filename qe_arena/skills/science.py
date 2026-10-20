"""natural-science families for qe_skill."""

from __future__ import annotations

import json
import random
from statistics import fmean
from typing import Any

from . import Built, Family
from ._common import close, load_object, rows, write_csv

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


FAMILIES = (
    Family(name="units", domain=DOMAIN, build=_units_build, check=_units_check, solve=_units_solve,
           traps={"keep_flagged": (1, 2, 3), "no_conversion": (2, 3), "nd_as_zero": (3,), "nd_dropped": (3,),
                  "all_analytes": (3,)}),
)
