# profa-qe-arena-v9r: predictions written before submission

Entry: v9 redraw. Dragan chose it on 2026-11-04 ("submit v9 redraw"). v9 itself (run `b302d037`, accepted
2026-10-29 10:11Z) was never evaluated: it is still `validated` with no evaluation entry, while v7 draw 8, accepted
a day later, was scored on 11-03. This request is `request-v9.json` with only `submission_id` and `name`
changed, so it is v9's first scored draw if it runs.

- request_sha256 (canonical): `0d61f5df4de57424fa40fab464984aba8b1fc2fded936b49a6b2aa23dea89155`; dry run passed.
  v9's own request still hashes to `bf140e34…`, so nothing else in the body changed.
- Image: `ghcr.io/proffesor-for-testing/openenv-qe-arena-terminal@sha256:228204ea…`, unchanged; the manifest
  resolves on ruos.
- Dataset: HEAD `29b93ada`, unchanged since 2026-10-29. The submit script does not submit if HEAD has moved.
- The cells, families and limits are v9's; see `predictions-v9.md`.

## Null model

From the leaderboard on 2026-11-04 at 14:30Z (254 `heldout-v2` runs by other accounts, 270 in all):

- A result that would be new for us: 20 of 254 (0.079).
- Science ≥ 1 or math ≥ 1: 9 of 254 (0.035).
- Industrial ≥ 3: 1 of 254.
- Totals: 2.85 ± 1.12 over all 270 runs, none above 5.
- Total ≤ 1: 26 of 254 (0.10), and 1 of our 16.

## Predictions

v9's, unchanged, so the series reads the same:

- **V1, new point:** at least one domain beats our banked best. Null 0.08; about 0.3–0.4 if the desk shape works
  as it appears to for pool-desk.
- **V2, science or math:** at least one science or math task solved. Null 0.035.
- **V3, total:** 2–5.
- **V4, no harm:** total ≥ 2. Null for a total of 1 or less: 0.10.

There is no training-health prediction.

## Reading rule

- This is draw 1 of v9's three-draw series. If v9 itself is later evaluated, it counts as a further draw.
- The series rule is `predictions-v9.md`'s: a new point in any draw, or a science or math solve in two, keeps the
  desk tier.
