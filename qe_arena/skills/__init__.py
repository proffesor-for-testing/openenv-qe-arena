"""qe_skill: content tasks (science, math, media) served only behind the terminal interface.

The arena's private tasks are Terminal-Bench 2 / SkillsBench-shaped: an instruction, input files in a working
directory, and hidden checks on an output file. These tasks train that work directly. A cell is a domain and a
level (``qe_skill-science-d2``); each seed draws one of the domain's families, as a code cell draws one corpus
function. Calibration can restrict a cell to one family with ``pool``; an episode cannot.

Every family is generated from ``(task_id, seed)`` by trusted code, which also computes the reference. Grading
reads the model's output file as data and never runs it, so no sandbox is needed. A family also ships a solver with
named wrong-method variants ("traps"), and generation retries until every trap that applies at the level fails at
least one check, so a seed never rewards a listed wrong method. Check names never reach an observation.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..domains import BY_SLUG, SLUGS

FAMILY = "qe_skill"
LEVELS = (1, 2, 3)
MAX_OUTPUT_BYTES = 65536
MAX_ATTEMPTS = 64  # generation retries per seed before a family gives up


@dataclass
class Built:
    """One generated task before it is bound to an ID: what the model sees and what grades it."""
    files: dict[str, str]
    statement: str
    output: str
    reference: Any


@dataclass(frozen=True)
class Family:
    name: str
    domain: str
    build: Callable[[int, random.Random], Built]
    check: Callable[[Any, str], list[bool]]
    solve: Callable[[dict[str, str], int, str | None], str]
    traps: dict[str, tuple[int, ...]]  # trap name -> levels where it applies


@dataclass
class SkillTask:
    task_id: str
    family: str
    difficulty: int
    seed: int
    domain: str
    kind: str
    prompt: str
    files: dict[str, str]
    output: str
    private: dict[str, Any] = field(repr=False)


def _families() -> dict[str, tuple[Family, ...]]:
    from . import formal, media, science  # noqa: PLC0415 - the modules import Built and Family from here

    by_domain: dict[str, list[Family]] = {}
    for module in (science, formal, media):
        for family in module.FAMILIES:
            by_domain.setdefault(family.domain, []).append(family)
    return {domain: tuple(fams) for domain, fams in by_domain.items()}


FAMILIES = _families()
ALL_SKILL_IDS = [f"{FAMILY}-{SLUGS[domain]}-d{level}" for domain in FAMILIES for level in LEVELS]

# What an episode may draw, per cell: the families whose strict untrained solve rate fell in 4-12 of 16 in the v7
# calibration (2026-10-20, runs/v7-cal; subtitles after its one adjustment round). Cells with no such family are
# not served. Calibration reaches any (cell, family) through ``pool``.
# v8 adds media-d3 from the same calibration, below that band on purpose: subtitles 3 and timecode 2 of 16, the two
# media families still unsolved most of the time (catalogue and assets, 13 of 16, stay out). It is last in
# ALL_SKILL_IDS, so every earlier cell keeps its position.
SERVED = {
    "qe_skill-science-d2": ("units", "calibration", "decay", "dilution"),
    "qe_skill-science-d3": ("units", "decay", "dilution"),
    "qe_skill-math-d1": ("linear", "counting", "congruences", "falsify"),
    "qe_skill-math-d2": ("counting", "congruences"),
    "qe_skill-math-d3": ("linear", "counting", "congruences"),
    "qe_skill-media-d1": ("timecode", "catalogue", "assets"),
    "qe_skill-media-d2": ("subtitles", "timecode", "catalogue", "assets"),
    "qe_skill-media-d3": ("subtitles", "timecode"),
}
SKILL_TASK_IDS = [task_id for task_id in ALL_SKILL_IDS if task_id in SERVED]


def parse_skill_id(task_id: str) -> tuple[str, int]:
    """(domain, level) of a qe_skill task ID (served, or reachable by calibration through a pool)."""
    if type(task_id) is not str or task_id not in ALL_SKILL_IDS:
        raise ValueError(f"invalid task_id: {task_id!r}")
    _, slug, level = task_id.split("-")
    return BY_SLUG[slug], int(level[1:])


def family_names(task_id: str) -> tuple[str, ...]:
    return tuple(f.name for f in FAMILIES[parse_skill_id(task_id)[0]])


def family(domain: str, name: str) -> Family:
    for fam in FAMILIES[domain]:
        if fam.name == name:
            return fam
    raise ValueError(f"no {name!r} family in {domain}")


def _trap_free(fam: Family, built: Built, level: int) -> bool:
    for trap, levels in fam.traps.items():
        if level in levels and all(fam.check(built.reference, fam.solve(built.files, level, trap))):
            return False
    return True


def make_skill_task(task_id: str, seed: int, *, pool: tuple[str, ...] | None = None) -> SkillTask:
    """Build the task for (task_id, seed). ``pool`` (family names) exists for calibration only."""
    domain, level = parse_skill_id(task_id)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64 bit integer")
    fams = FAMILIES[domain]
    if pool is not None:
        if not pool or any(name not in {f.name for f in fams} for name in pool):
            raise ValueError(f"pool must name {domain} families: {[f.name for f in fams]}")
        fams = tuple(f for f in fams if f.name in pool)
    elif task_id in SERVED:
        fams = tuple(f for f in fams if f.name in SERVED[task_id])
    else:
        raise ValueError(f"{task_id} is not served (no family passed calibration); calibration must name a pool")
    rng = random.Random(f"{task_id}:{seed}")
    fam = rng.choice(fams)
    for _ in range(MAX_ATTEMPTS):
        built = fam.build(level, rng)
        if _trap_free(fam, built, level):
            break
    else:
        raise RuntimeError(f"{fam.name} could not build a trap-free task for {task_id} seed {seed}")
    return SkillTask(task_id=task_id, family=FAMILY, difficulty=level, seed=seed, domain=domain, kind=fam.name,
                     prompt=built.statement, files=built.files, output=built.output,
                     private={"reference": built.reference, "family": fam, "corpus": fam.name})  # calibration's key


def grade(task: SkillTask, text: str | None) -> float:
    """1.0 only if every check passes on the output file; a missing, oversized or unparseable file scores 0."""
    if text is None or len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        return 0.0
    try:
        results = task.private["family"].check(task.private["reference"], text)
    except Exception:  # noqa: BLE001 - malformed model output is a wrong answer, never a crash
        return 0.0
    return 1.0 if results and all(results) else 0.0


def solve(task: SkillTask, variant: str | None = None) -> str:
    """The oracle's output (variant None) or a named wrong method's, computed from the visible files only."""
    return task.private["family"].solve(task.files, task.difficulty, variant)
