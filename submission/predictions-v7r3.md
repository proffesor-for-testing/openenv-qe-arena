# profa-qe-arena-v7r3: predictions written before submission

Entry: v7 draw 4. The request is v7's, unchanged apart from the id and name. It has the same seven `qe_skill`
content cells (science d2 and d3; math d1, d2 and d3; media d1 and d2), the same image
`terminal@sha256:9307e4e6…` and the same limits. The request names no dataset revision, and the dataset repo's HEAD
is still `f5e0b95`. No upload to that repo will be made until this run has finished training.

- request_sha256 (canonical): `202260069f124b46b9eacd00423284503af878dc50f16a50e5375a04cf9a0df9`
- `request-v7r3.json` file sha256: `477c2889…`

## Why

Draws 1–3 met the three-draw rule set in `predictions-v7.md`: 2 of 3 runs had a science, math or media solve. Both
solves were one media task, and neither science nor math was solved. Media was singled out only after the draws were
seen (`result-v7r2.md`), so it has not yet been tested. This series tests it on fresh draws.

Each draw is also a chance at a new per-domain best. Media is banked at 1, so the content cells can add a point
only through media ≥ 2, natural-science ≥ 1 or mathematics ≥ 1.

## Null model

Leaderboard snapshot of 2026-10-23 at 12:47Z, 184 evaluated runs:

| Quantity | Count | Rate per run |
|---|---|---|
| Media solve, other accounts only | 4 of 174 | **0.023** |
| Any new per-domain best for us (beats media 1, industrial 2, office 2, software 1 or finance 1, or solves any other domain) | 15 of 184 | **0.082** |
| Total per run | — | 2.84 ± 1.11 |
| Media ≥ 2 | 0 of 184 | — |

## Predictions

- **M1, media (the test):** at least one media task solved. P = 0.023 under the null. If our three draws reflect a
  real rate, it is about 0.5–0.67.
- **M2, new point (descriptive):** P ≈ 0.08 under the null. If the media lift is real and is about 0.15–0.2 per
  task, media ≥ 2 alone has P ≈ 0.16–0.26.
- **M3, total:** 2–4, the null band.

There is **no training-health prediction.** Draw 3 scored our best with saturated cells, and update counts do not
predict scores on the board (#300). Trackio metrics are reported as description only.

## Reading rule

**This draw on its own:**
- A media solve counts as evidence for a media lift (P ≈ 0.023 under the null).
- A miss is weak evidence against: P ≈ 0.33–0.5 at the hypothesised rate.

**Series:** up to three new draws, r3 to r5. Each needs its own approval of its exact sha.
- 2 or more of 3 with a media solve confirms the lift. Under the null that is P ≈ 0.0016; at a rate of 0.5 it is
  P = 0.5.
- 0 of 3 rejects it: P = 0.125 at a rate of 0.5.
- 1 of 3 is inconclusive.
- The series stops early once confirmed or rejected, or once v8 is ready and approved.

Not changed by this file: the earlier three-draw rule and its result stand as written in `result-v7r2.md`.
