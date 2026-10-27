# v9 design: desk tasks on real public data

Status: in progress on branch `feat/v9-desk`, started 2026-10-27 on Dragan's "start option B". Four families are built
and tested; nothing is calibrated, served or submitted. Session cognitum-d0.

## Why v9

The v7 request has run its course: 7 draws (6 ours, 1 NoeFlandre's redraw) solved media 4 times and added no
composite point (`submission/result-v7r5.md`). The leaderboard and the other entrants' public datasets point to a
different lever (research of 2026-10-27, 229 runs, 13 dataset revisions read):

- **Domain content does not seem to carry over to the held-out set.** No science, math or cybersecurity hit has
  repeated on a redraw of the same dataset. Some hits came from runs that trained none of that domain (pool-desk-v6
  scored office 3 with no office tasks; terminal-desk scored math 1 with no math tasks). Runs that trained a domain
  heavily got nothing in it (science-desk: 12 science tasks, science 0; our v7: science and math in every draw,
  never solved). Our media lift is the one repeatable content effect on the board, and it stops at one task.
- **What the base model loses on the held-out tasks is mostly the finish.** In dnakhla's test on public SkillsBench
  tasks under arena budgets, 46 of 47 failures ended before any answer was graded. On pool-desk-v6's 30 calibration
  measurements, 162 attempts broke the reply format and 97 ran out of budget; 50 finished.
- **Only one environment family beats the field's rate of results that would be new for us:** jjmachan's pool-desk,
  5 of 11 runs against 0.079 (binomial P 0.001; about 0.02 after correcting for 22 families, grouping chosen after
  looking). pool-desk-v6 averages 4.0 over 5 runs; the field averages 2.86 ± 1.11, below the untrained 4/40.
- **What pool-desk does:** Terminal-Bench-2 task folders on pinned slices of real public data; long instructions
  (median about 1,800 characters); inputs too large to print; method rules with real traps; one output file at a
  fixed path with a fixed header; binary pytest verifiers; tasks admitted only when the base model solves 25–75%.

**Hypothesis:** training on eval-shaped work (long brief, big real file, script it, write one exact file, finish in
budget) lifts the run's general finishing rate, and with it every domain's chance, including the ones we have never
scored. This is an inference from observational data. v9 tests it with our own environment rather than copying
pool-desk's tasks.

## What a desk family is

A new tier of the existing `qe_skill` family: level 4 (`qe_skill-<slug>-d4`), appended after the v7 cells so every
earlier cell keeps its ID and position. Each desk family:

- reads a **pinned snapshot of real public data** from `qe_arena/skills/data/` (source URL, licence, retrieval date
  and SHA-256 in `PROVENANCE.json`; `scripts/fetch_desk_data.py` re-fetches, `--check` re-hashes). Generation never
  touches the network;
- writes the **real file** into the workspace (or a seeded window of it), with its own headers, codes and quirks,
  plus a `request.json` of seeded settings. The largest input exceeds 16 KiB, twice the 8 KiB at which a command's
  output is cut, so the model must script;
- states the work as a **long prose brief** (1,200–2,000 characters) with method rules and one exact CSV contract;
- is graded by **named checks on the output file** read as data, all-or-nothing, as in v7;
- ships a **solver with traps**. Generation retries until every listed trap fails. Request dates and windows are
  chosen so that the rules bind on the real data rather than being decorative (for example, the KEV window ends on
  a day entries were added).

Only US-government public-domain data or synthetic data are used, so no attribution or share-alike terms constrain
reuse.

## Families

| Domain | Family | Data | Brief | Status |
|---|---|---|---|---|
| natural-science | `streamflow` | USGS NWIS daily discharge, 10 gauges, WY2015–2026 (A, A:e, A:R, P, Ice) | per water year: approved days, mean, N-day low flow, peak date; unit conversion; completeness rule | **built, tested** |
| cybersecurity | `kev` | CISA Known Exploited Vulnerabilities catalogue, 1,758 entries | vendor exposure report: window, ransomware, overdue, median days to due, top CWE with numeric tie-break | **built, tested** |
| finance-economics | `yields` | US Treasury daily par yield curve files, 2018–2026 as published (newest first, MM/DD/YYYY, columns change by year, blank tenors) | inversion spells of a tenor spread in bp across files, skipped days, minimum spell length | **built, tested** |
| office-white-collar | `calendar` | synthetic ICS export (RRULE, EXDATE, time zones) | meeting hours per person and week, conflicts | planned |
| mathematics-or-formal-reasoning | `ledger` | synthetic 650–850-row ledger: six notations (integer, decimal, fraction, mixed number, repeating decimal, scientific), corrections, void rows | exact per-group totals as reduced fractions | **built, tested** |
| industrial-physical-systems | `quakes` or `grid` | USGS earthquake catalogue or EIA-930 hourly demand | event declustering or demand peaks with timezone and gap rules | planned |
| cybersecurity | `authlog` | synthetic sshd/auth.log | brute-force windows, lockouts, successful-after-failure | planned |
| natural-science | a second family | NOAA GML greenhouse-gas monthly series | de-seasonalised trend with missing-value codes | planned |

Media and software are left out: media is banked at 1 and has never reached 2 on the board, and software's only
solves look like background noise.

## Verifier tests

The v7 suite covers desk families automatically through `ALL_SKILL_IDS`: the oracle earns 1.0 on 200 seeds; every
trap scores 0 on every seed; empty, copied and malformed outputs score 0; nothing leaks into the workspace or the
instruction; seeds rebuild identically. `tests/test_skills_desk.py` adds: pinned data matches its provenance;
desk cells are not served before calibration; briefs are long and inputs too big to print; stated rounding is
accepted and anything beyond it rejected; incomplete years must stay empty; row order and header rules; Ice and
provisional values are never approved; vendor names are trimmed; CWE ties go to the lowest number.

## Calibration plan

Same tooling and regime as v7 (`scripts/calibrate.py --interface terminal --strict`, Qwen/Qwen3.8-27B with
thinking off through the HF router, in the terminal image, uid mode on ruos): 16 strict episodes per family, keep
a family when it solves 4–12 of 16 with budget endings under 25%, one adjustment round, then drop. Desk episodes
are longer than v7's, so the token cap needs re-estimating once all families exist; about 3–6M tokens for 8
families is the first guess. **Dragan approves the cap before the run.**

## Request

One `qe_skill-<slug>-d4` cell per desk domain that passes calibration, in the terminal image with a new digest,
on a dataset revision exported with the desk rows. Predictions are committed before submission, as before.

## Risks

- The mechanism is inferred, and the pool-desk effect could still be selection or noise (5 of 11; v6 alone 2 of 5).
- Real-data desk tasks may calibrate below the band: the base model solves about 3 of 50 public SkillsBench tasks
  under arena budgets. The adjustment round can only make a family easier (fewer rules, smaller files).
- Our briefs are written to SkillsBench's shape but not from its tasks; if the eval's shape differs, the lever is
  weaker.
