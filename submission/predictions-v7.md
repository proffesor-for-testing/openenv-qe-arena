# profa-qe-arena-v7: predictions written before submission

Entry: "QE content". The terminal interface is unchanged (`exec` / `write_file` / `evaluate`, 4096 completion and
8192 context tokens, 32 calls). The train cells are the 7 served `qe_skill` content cells: science, math and media
tasks with input files, graded by hidden checks on one output file (`docs/v7-domain-tasks.md`). Each cell draws
only the families whose strict untrained solve rate was 4–12 of 16 in the v7 calibration:

| Domain | Cell | Families served |
|---|---|---|
| natural-science | qe_skill-science-d2 | units, calibration, decay, dilution |
| natural-science | qe_skill-science-d3 | units, decay, dilution |
| mathematics | qe_skill-math-d1 | linear, counting, congruences, falsify |
| mathematics | qe_skill-math-d2 | counting, congruences |
| mathematics | qe_skill-math-d3 | linear, counting, congruences |
| media | qe_skill-media-d1 | timecode, catalogue, assets |
| media | qe_skill-media-d2 | subtitles, timecode, catalogue, assets |

Image: `ghcr.io/proffesor-for-testing/openenv-qe-arena-terminal@sha256:9307e4e6…` (main `7490e2d`). Dataset: `Profa-QE/openenv-qe-arena-tasks` revision `f5e0b95` (448 skill rows, 64 seeds per cell). Request sha256 `6aef0b10…`.

## Why
- On the board, shell environments that train content work have about 0.23 runs with a science, math or media solve
  per run. Everything else has about 0.05. Domain-targeted environments did not hit their target domains
  (Syndicate604's math and science desks scored 0 there). So v7 tests the kind of task, not the targeting.
- Our 7 runs so far, all code QE, solved science, math and media 0 times. Our composite is 6/40 and can only rise
  through a new domain.

## Null model
Across 143 evaluated board runs, 10 had at least one solve in science, math or media: **0.07 per run**. Totals are
2.8 ± 1.1 per run (#275, #294). v7 has no industrial or office train cell, and banked bests cannot fall, so those
domains matter only to the total, not to the composite.

## Predictions
- **P1, training signal (Trackio, if the API answers):** mean reward rises from the first quarter to the last, and
  the share of steps with a nonzero gradient is ≥ 40%. The cells were calibrated to mixed solve rates, so this
  should hold.
- **P2, total:** 2–4 / 40, the null band.
- **P3, the lever:** at least one solve in science, math or media.
  - Under the null, P = 0.07 per run. At the content-shell rate, about 0.23.
  - One run cannot separate the two, so **three v7 draws are planned**. Under the null, 2 or more of 3 runs with
    such a solve has P ≈ 0.014; at 0.23 per run it is ≈ 0.13. That count over three runs is the evidence.
- **P4, composite:** rises to ≥ 7/40 if and only if P3 holds in a domain we have not solved. All three qualify.
- **P5, harm:** 0–1 / 40, the level of zero-update runs (#295). That would mean the content cells hurt the evaluation.

## Reading rule
- One run reads as inconclusive whatever it scores. A P3 hit is a composite gain, not yet evidence for the lever.
- After three draws: 2 or more runs with a science, math or media solve counts as evidence for content training.
  0 counts as weak evidence against at this sample size (P ≈ 0.46 even at 0.23 per run).
- If P1 fails, the run tells nothing about the lever.
