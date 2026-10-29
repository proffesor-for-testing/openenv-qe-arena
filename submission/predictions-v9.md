# profa-qe-arena-v9: predictions written before submission

Entry: v9, the desk tier (`docs/v9-desk-tasks.md`, merged in `4187c74`). Dragan chose it on 2026-10-29 for the
slot that opens about 09:30:34Z, on condition that it was ready in time ("if it is ready we will go with it, if not
ready, another v7 draw").

The request trains three `qe_skill-<domain>-d4` cells in the terminal image. Each cell is one desk family that the
2026-10-28 calibration kept (4–12 of 16 solved, no more than 3 of 16 ending on the budget):

| Cell | Family | Data | Calibration |
|---|---|---|---|
| `qe_skill-science-d4` | ghg | NOAA GML in-situ monthly CO2/CH4 | 8/16, 3 budget endings |
| `qe_skill-math-d4` | ledger | synthetic ledger, exact fractions | 6/16, 2 |
| `qe_skill-industrial-d4` | quakes | USGS ComCat catalogue slices | 9/16, 3 |

The limits are v7's (4096 completion tokens, 8192 context, 32 tool calls), which are the budgets the calibration
used.

- request_sha256 (canonical): `bf140e349e7d9bdedd968e5d4e1cb6a7962fbe26831d1ab187b3572b59192e05`; dry run passed.
- Image: `ghcr.io/proffesor-for-testing/openenv-qe-arena-terminal@sha256:228204ea…` (main `a570c25`; CI run
  40838199638 green, slow shards included). Pulled anonymously on ruos; smoke 59/59 task episodes, uid mode; live
  `/schema` equals the request's.
- Dataset: HEAD `29b93ada` (exported from `a570c25`). It adds 64 rows for each d4 cell; every earlier row and the
  other two configs are byte-identical to `99808a44`. The submit script does not submit if HEAD has moved.
- Submitted after the slot opened (09:30:34Z): the dataset export took 2 h 10 min on ruos.

## Why this request

The research of 2026-10-27 found one environment family above the field's rate of results that would be new for
us: jjmachan's pool-desk (long briefs over real data, one exact output). Its 12 runs on the board average 3.5/40
against the field's 2.85. v9 copies that shape, not its tasks. Content targeting has not carried over to the
held-out set in any account's runs, ours included, so the cells' domains are not expected to be the domains that
score.

## Null model

From the leaderboard on 2026-10-29 at 07:50Z (225 `heldout-v2` runs by other accounts):

- A result that would be new for us: 18 of 225 (0.080).
- Science ≥ 1 or math ≥ 1: 8 of 225 (0.036).
- Industrial ≥ 2 (our banked best): 66 of 225 (0.29); industrial ≥ 3: 1 of 225.
- Totals: 2.85 ± 1.10 over all 240 runs, none above 5.

## Predictions

- **V1, new point:** at least one domain beats our banked best. Null 0.08. If the desk shape works as it appears
  to for pool-desk, about 0.3–0.4. This is the only prediction that can change the composite.
- **V2, science or math:** at least one science or math task solved. Null 0.036.
- **V3, total:** 2–5.
- **V4, no harm:** total ≥ 2. A total of 1 or less came up in 21 of 225 runs by others (0.093) and in 1 of our 15.

There is no training-health prediction. Trackio metrics are reported as description only.

## Reading rule

- One run is noise at these rates. v9 is planned as a series of three draws of the same request.
- If any of the three draws gives a new point (V1), or two give a science or math solve (V2), the desk tier is
  kept and extended with the families calibration dropped. If none of the three does either, the desk tier is
  judged no better than the field and the plan is reopened.
