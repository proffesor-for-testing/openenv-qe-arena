# profa-qe-arena-v7r5: predictions written before submission

Entry: v7 draw 6. The request is `request-v7r3.json` with only `submission_id` and `name` changed: the same seven
`qe_skill` content cells (science d2 and d3; math d1, d2 and d3; media d1 and d2), the same image
`terminal@sha256:9307e4e6…` and the same limits. The media series closed at 2 of 2 (`result-v7r4.md`), so this draw
is not part of it.

- request_sha256 (canonical): `e8eea61afd7d97ddcc2ddc60466863ac29c0bcfbb5e56e869acbc0a2ba4152af`
- `request-v7r5.json` file sha256: `e61b8b95…`
- Dry run passed on 2026-10-26.
- Dataset: HEAD `99808a44`, as in r4. The submit script does not submit if HEAD has moved.

## Why this request

It is a free draw on the only request of ours with repeat content solves (media in 4 of 5 draws, totals 2, 1, 5, 2
and 4). No other request we have can do better without new code. It can add a composite point only through
media ≥ 2, natural-science ≥ 1, math ≥ 1, or a domain it does not train (office ≥ 3, industrial ≥ 3, finance ≥ 2,
software ≥ 2, cybersecurity ≥ 1).

## Null model

Leaderboard of 2026-10-26 at 14:30Z, 217 evaluated runs:

| Quantity | Count | Rate per run |
|---|---|---|
| Media solve, other accounts only | 4 of 204 | 0.020 |
| Any new per-domain best for us | 17 of 204 other-account runs | 0.083 |
| Total per run | — | 2.86 ± 1.11 |
| Media ≥ 2 | 0 of 217 | — |

## Predictions

- **R1, media:** at least one media task solved. This is descriptive now: about 0.8 on our record, 0.020 under the
  null.
- **R2, new point:** P ≈ 0.08 under the null. Our five v7 draws produced none, so we do not expect one.
- **R3, total:** 2–5, our v7 range.

There is **no training-health prediction.** Trackio metrics are reported as description only.

## Reading rule

- A new point is the only result that changes the composite.
- A media miss does not reopen the series. It is reported as the sixth draw of the request.
