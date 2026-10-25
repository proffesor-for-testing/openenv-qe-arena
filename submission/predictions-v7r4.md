# profa-qe-arena-v7r4: predictions written before submission

Entry: v7 draw 5, and draw 2 of the media series (r3 to r5). The request is `request-v7r3.json` with only
`submission_id` and `name` changed. It has the same seven `qe_skill` content cells (science d2 and d3; math d1, d2
and d3; media d1 and d2), the same image `terminal@sha256:9307e4e6…` and the same limits.

- request_sha256 (canonical): `9e282c52426c300537d6c506257c8335da6ef20486c8f557651596b0da1562be`
- `request-v7r4.json` file sha256: `d2c73189…`
- Dry run passed on 2026-10-25.

## Dataset

The request names no dataset revision. HEAD is `99808a44`, the v8 upload. It differs from `f5e0b952` (draws 1–4)
only by 64 added `qe_skill-media-d3` rows, a cell this request does not name; every other row is byte-identical.
`submit-v7r4.sh` does not submit if HEAD has moved from `99808a44`.

## Series status

The series stands at 1 of 1 (r3 solved media). `predictions-v7r3.md` said the series stops early once v8 is ready
and approved. v8 ran and did not solve media (`result-v8.md`), and its reading rule defers the v8-or-v7 choice to the
remaining r-series draws. On 2026-10-25 Dragan chose to continue the series with r4. That clause is waived; the
confirm and reject thresholds are unchanged.

## Null model

Leaderboard snapshot of 2026-10-25 at 15:00Z, 207 evaluated runs:

| Quantity | Count | Rate per run |
|---|---|---|
| Media solve, other accounts only | 4 of 195 | **0.021** |
| Any new per-domain best for us (beats media 1, industrial 2, office 2, software 1 or finance 1, or solves any other domain) | 15 of 207 | **0.072** |
| Total per run | — | 2.86 ± 1.12 |
| Media ≥ 2 | 0 of 207 | — |

## Predictions

- **M1, media (the test):** at least one media task solved. P = 0.021 under the null. Our v7 draws (3 of 4 with a
  media solve) suggest about 0.5–0.75.
- **M2, new point (descriptive):** P ≈ 0.07 under the null.
- **M3, total:** 2–4, the null band.

There is **no training-health prediction.** Trackio metrics are reported as description only.

## Reading rule

- A media solve here makes the series 2 of 2 and **confirms** the lift (2 or more of 3). r5 is then not needed for
  the series.
- A miss makes it 1 of 2; r5 decides (2 of 3 confirms, 1 of 3 is inconclusive).
- Any further draw needs its own approval of its exact sha.
