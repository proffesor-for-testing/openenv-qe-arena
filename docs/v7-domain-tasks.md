# v7 design: content tasks for science, math and media behind the terminal interface

Status: draft for Dragan's decision, 2026-10-20. Nothing is built or submitted. Session cognitum-d0.

## Goal and the evidence behind it

Our banked composite is 6/40 (software 1, finance 1, industrial 2, office 2). It can only rise through a domain we have
never solved. Office and industrial are already at the board's usual ceiling. Every Profa-QE run so far trained code
QE work (bug repair, mutant kill), and none of our 7 runs solved science, math or media.

What the board shows (`/api/leaderboard`, 143 evaluated runs, read 2026-10-20):

- **Rare domains are rare for everyone.** Science 5 solves, media 3, math 2 and cybersecurity 1 across all runs.
  10 different submissions produced them, and none produced two.
- **Targeting a domain did not produce that domain.** Syndicate604's math-targeted (`evidence-desk-math-v3`, `-v4`,
  `-v4-rerun`) and science-targeted (`science-desk-v5`) runs scored 0 math and 0 science. Their one science solve
  came from the general `skills-desk-v1`.
- **The environment type is the one pattern.** Shell environments that train content work (data files, calculations,
  formatted outputs in SkillsBench prose: dnakhla terminal-desk, Syndicate604 desks, jjmachan pool-desk, naklitechie
  terminal-habits) have 5 rare-domain solves in 22 runs, about 0.23 per run. Other environments have about 0.04 per
  run. NoeFlandre's #304 found the same split (5 of 7 science/math solves from 24 shell runs, one-sided Fisher
  p ≈ 0.0015). The split was drawn after looking at the data, and n is 7 events.
- **Our terminal runs are shell-style but code-only**, and scored 0 of 4 (about 0.9 expected at 0.23 per run, so this
  is not evidence against the pattern either way).

**Hypothesis for v7:** content tasks in a shell raise a run's chance of a rare-domain solve from about 0.05 to about
0.2. This is a hypothesis about task kind, not about domain targeting. A rare-domain point is worth +1 composite,
so the expected value is about +0.15 composite per run over a code-only draw. It is a modest, uncertain lever, and
v7 is worth building only because each run is otherwise a free draw.

The private evaluation is Terminal-Bench 2 / SkillsBench-shaped (#196, #197, dnakhla's card). A task arrives as an
instruction plus files in a workspace. The agent uses `exec`, `write_file` and `evaluate`; hidden checks read output
files; there are 4096 completion tokens, 8192 context tokens and 32 tool calls. Our terminal image already serves
that contract (`qe_arena/terminal_env.py`), so v7 adds task families, not a new interface.

## What v7 adds

A new family, `qe_skill`, with cells `qe_skill-<slug>-d<level>` (slugs `science`, `math`, `media`; levels 1–3). It is
served only by the terminal interface and generated procedurally from `(task_id, seed)`, like every existing cell.
New IDs are appended to `ALL_TASK_IDS`, so existing cells and their 24 + 4 golden pins stay byte-identical.

Each task:
- writes **input files** into the workspace (CSV, JSON, SRT, plain text), never the reference;
- states the work in SkillsBench-style prose with one exact output contract: path, format, rounding and units;
- is graded at `evaluate` by **named checks** on the output file. The reward is 1 only if every check passes;
  check names are never shown;
- has a **reference computed by trusted generator code** in the server process. Grading reads data, never runs model
  code, so the WASM sandbox is not needed. Output files are size-capped (64 KiB) and parsed strictly.

QE is the part we bring that others do not. Each family is built like our mutant-kill cells: every plausible wrong
method must fail at least one check, on a stated share of seeds (see "Verifier tests").

## Task families

Four families per domain; levels add steps, not obscurity. "Traps" are the wrong methods the checks must reject.

### Science (`qe_skill-science-*`)
| Family | Input → output | d1 → d3 | Traps the checks reject |
|---|---|---|---|
| S1 unit-aware aggregation | measurement CSV (sample, analyte, value, unit, QC flag) → per-sample mean in a requested unit, JSON | one unit → mixed mg/L, µg/mL, ppm → plus below-detection rows substituted at LOD/2 | no conversion; flagged rows kept; BDL as 0 |
| S2 calibration curve | standards (conc, absorbance) + unknowns → slope, intercept, R², unknown concentrations | plain fit → blank subtraction → one outlier standard dropped by a stated residual rule | no blank; fit through origin; outlier kept |
| S3 decay / half-life | count time series → half-life (tolerance 1%) | one source → background subtraction → background plus dead-time correction | linear fit on raw counts; background ignored |
| S4 dilution plan | stock concentrations, targets → volumes CSV with rounding rule | single dilution → serial chain → minimum pipette volume constraint | C1V1 inverted; rounding before chaining |

### Math (`qe_skill-math-*`)
| Family | Input → output | d1 → d3 | Traps |
|---|---|---|---|
| M1 exact linear algebra | integer system → exact solution as `p/q` strings | 2×2 → 3×3 → 4×4 with a singular case reported as `"none"` / free variables | floats; unreduced fractions; singular case missed |
| M2 constrained counting | arrangement rules → integer count (generator verifies by brute force) | one constraint → two → inclusion–exclusion over three | ignoring overlap; counting order twice |
| M3 congruences | system of congruences → least non-negative solution and modulus | coprime CRT → non-coprime → inconsistent system reported | assuming coprime; not reducing |
| M4 falsify a claim | a claimed identity or inequality and a search range → smallest counterexample or `"holds"` | single variable → two variables → claim true on the range | off-by-one on range; first found ≠ smallest |

### Media (`qe_skill-media-*`)
| Family | Input → output | d1 → d3 | Traps |
|---|---|---|---|
| D1 subtitle repair | SRT with timing faults → corrected SRT to a spec | offset shift → frame-rate conversion (23.976 → 25) → plus min gap and max chars per line | rounding per cue drift; overlaps left; renumbering skipped |
| D2 timecode / EDL | edit list in SMPTE timecode at an fps → total runtime and per-clip frames | non-drop 25 fps → 30 fps → 29.97 drop-frame | treating drop-frame as non-drop; off-by-one frame |
| D3 catalogue normalisation | media library CSV → JSON with ISO 8601 durations, deduped IDs, title rules | durations → plus dedupe → plus a playlist under a runtime cap with a tie-break | duplicates kept; cap exceeded; tie-break ignored |
| D4 asset specs | source dimensions and platform specs → crop/resize table | one aspect ratio → several → safe-area margins with even-pixel rounding | aspect distorted; odd dimensions |

## Verifier tests (built before any calibration)

Per family, in `tests/test_skill_<domain>.py`, modelled on `tests/test_d4_attacks.py`:
- **Oracle:** the reference output earns 1.0 on 200 seeds within 32 actions.
- **Negative controls:** an empty file, a copy of the input, and the wrong format all score 0 on every seed.
- **Traps:** each listed wrong method is implemented and scores 0 on at least 95% of seeds. A trap that passes more
  than 5% of seeds means its check is too weak, and the family is not served until that check is fixed.
- **Tolerance:** the oracle's output, perturbed within the stated tolerance or rounding rule, still passes. Values
  perturbed just outside it fail.
- **No leak:** the reference and check names never appear in the workspace, the instruction or an observation.
- **Golden pins** for each new cell, so generation stays reproducible, following `runtime_pins.json` and the
  signed-manifest flow.

Before calibration, one more check: the episode shell's `python3` in the terminal image has numpy and pandas
(`requirements.lock` lists both). Confirm they import as the episode uid.

## Calibration plan

The same tooling and regime as v5: `scripts/calibrate.py --interface terminal --strict`, Qwen/Qwen3.8-27B with
thinking off through the HF router, run inside the terminal image as root on ruos (uid isolation is required).

1. **Screen:** 12 families × 3 levels = 36 cells. 4 seeds × 4 attempts = 16 episodes per cell, 576 episodes in total.
   The v5 calibration averaged about 2,000 peak context tokens and 4.8 actions per episode, so a cap of
   `--max-total-tokens 8000000` should cover it. The cost depends on router pricing. **Dragan approves the cap before
   the run.**
2. **Keep rule** (the arena's guidance band, #263): a cell is kept if its strict solve rate is 4–12 of 16
   (0.25–0.75), its completion-budget endings are under 25%, and its protocol failures are under 15%.
3. **One adjustment round:** a family with no level in the band gets its level parameters (rows, steps, constraint
   count) moved once and is screened again (16 more episodes per cell). If it still misses, it is dropped. No third
   round, so the calibration cannot be tuned until something happens to pass.
4. **Record:** every row is written to `runs/v7-cal/*.jsonl` with the summary in the predictions file, as in v5.

## The v7 request (after calibration)

- **Same image contract** with a new digest (the families are new code). The dataset is a new revision exported with
  the `qe_skill` rows.
- **8 train cells, all content:** about 3 science, 3 math and 2 media, chosen from the kept cells, at most one cell
  per family. Industrial and office cells are left out: they cannot raise the composite.
- **Predictions are committed before submission**, in the v5/v6 format:
  - P1: training signal, with reward rising and gradient share in a stated band;
  - P2: total 2–4;
  - P3: a solve in science, math or media. The null is about 0.05 per run. Content-shell envs run at about 0.23.
- **Reading rule:** one run cannot separate 0.05 from 0.23. Plan three v7 draws (three days). Under the null the
  chance of 2 or more rare-domain solves in 3 runs is about 0.01; at 0.23 per run it is about 0.13. The plan says so
  up front, so a single lucky or unlucky draw is not over-read.

## Effort and order

| Step | Work | Estimate |
|---|---|---|
| 1 | Family scaffolding (`qe_skill`, serving, export, pins); S1, M1, D1 with full verifier tests | 1 day |
| 2 | The remaining 9 families with their verifier tests; CI | 1–1.5 days |
| 3 | Image and dataset build; calibration screen; one adjustment round | 1 day (screen is mostly wall-clock) |
| 4 | Request, predictions, Dragan's sha approval, submission | 0.5 day |

Meanwhile the daily slot can still take a free draw (a v6 or v5 replicate), each with Dragan's sha approval.

## Risks and open questions

- **The evidence is observational:** 7 events and a split drawn after the fact. Targeted envs (Syndicate604) did not
  hit their target domains. v7 may simply score like v5 and v6.
- **We do not know the private science, math and media tasks.** The families above are guesses from SkillsBench's
  shape, not from the hidden tasks.
- **Parity with the operator's TB2 schema:** dnakhla repeats the instruction in every observation and serves a `close`
  action. Ours shows the instruction at reset only. Should v7 match that? It would change the observations of
  existing cells, so it would need new pins and a separate decision.
- **Grading outside WASM is new for us.** It reads data only, but it needs its own review: parser limits, no `eval`,
  no YAML or pickle.
