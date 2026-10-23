# profa-qe-arena-v8: predictions written before submission

Entry: v8, the media-weighted successor to v7. It trains five `qe_skill` content cells: science d2 and d3, and
media d1, d2 and d3. Math is dropped from the request: none of our three v7 draws solved a math task, and 2 of 184
board runs have. Media d3 is newly served (PR #5, `1b2b4fa`). It uses the two media families that the v7
calibration found hardest for the untrained model: subtitles at 3 of 16 and timecode at 2 of 16 strict solves
(`docs/v7-domain-tasks.md`). The other cells, the limits and the action schema are v7's.

- Image: `terminal@sha256:19a7938a…` (main `1b2b4fa`, CI run 39968807839, all jobs green). It serves all eight
  cells, including `qe_skill-media-d3`.
- request_sha256 (canonical): `535b668b3f1ca7669819e507e0ba438119c04833f5bbef875006e7ac2f3162da`
- `request-v8.json` file sha256: `35b83656…`
- Dry run passed on 2026-10-23.

## Dataset order

The request names no dataset revision, so the arena may read the dataset repo's HEAD. The media-d3 rows will be
exported only after v7r3 (draw 4) has finished training, about 2026-10-24 at 12:35Z, so that draw 4 still trains on
`f5e0b95`. v8 is not submitted before the new revision is on the hub. Its hash goes in the result file.

## Why

- Media is the only content domain we have solved: 2 of 3 v7 draws, against 4 of 174 runs on other accounts.
- Late in v7r2 the cells were saturated. The share of groups with zero reward variance rose from 0.48 to 0.73, and
  most of those groups were all-solved. Media d3 sits below that band, so it keeps a learning signal late in the
  run.
- The cells cut from the request are the ones with no result. Nothing new was calibrated, and the reward is still
  v7's strict pass/fail. Partial-credit reward was left out: nothing on the board shows it moves the score.

## Null model

The null is the same as in `predictions-v7r3.md`, from the leaderboard of 2026-10-23 at 12:47Z (184 evaluated runs).

| Quantity | Rate per run |
|---|---|
| Media solve, other accounts only | 0.023 |
| Any new per-domain best for us | 0.082 |
| Total per run | 2.84 ± 1.11 |
| Media ≥ 2 | 0 of 184 |

## Predictions

- **V1, media:** at least one media task solved. P = 0.023 under the null. Our v7 draws suggest about 0.5–0.67.
- **V2, media ≥ 2 (the reason for v8):** P ≈ 0 under the null, since no run on the board has done it. If v8 lifts
  the per-task media rate to about 0.2–0.3, P ≈ 0.26–0.47. That would be a new point for the composite (7 → 8).
- **V3, total:** 2–4, the null band.
- **V4, no loss elsewhere:** science, math and every other domain stay within the v5–v7r2 range (0–2 per domain).
  Math at 0 is expected and is not a miss, since none of our runs has solved it.

There is **no training-health prediction.** Trackio metrics are reported as description only.

## Reading rule

- A V2 hit is the result that matters. It raises the composite, and it is strong evidence for media-weighted
  training on its own.
- A V1 hit without V2 counts toward the media lift in the same way as an r-series draw. It is reported beside that
  series, not counted inside it, because the request differs.
- A V1 miss is weak evidence against: P ≈ 0.33–0.5 at the hypothesised rate.
- Decide on continuing v8 or reverting to v7 after one v8 draw and the remaining r-series draws.
