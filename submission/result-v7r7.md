# profa-qe-arena-v7r7: result, read against predictions-v7r7.md

Draw 8 of the v7 request on our account. The request body is v7's apart from the id and name (request_sha256
`091a8d87…`). The predictions were committed in `601b53a` before the submission, and the receipt in `4e03298`.

The run was accepted at 10:12:18Z on 2026-10-30 as run `8ebea8b35960ab7707030a7f` and evaluated at 08:52Z on
2026-11-03 on `heldout-v2` after 180 steps. The leaderboard records dataset revision `29b93ada`, so the dataset
guard held.

The arena's training cluster was unavailable from about 10-30 22:30Z until 11-03 (503 `UPSTREAM_UNAVAILABLE`, as
rca-umb also reported in #425). Our run was queued at 13:45Z on 10-30, before the outage, and scored when it ended.

| Domain | v7 | v7r | v7r2 | v7r3 | v7r4 | v7r5 | v7r6 | v7r7 |
|---|---|---|---|---|---|---|---|---|
| industrial-physical-systems | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 0 |
| office-white-collar | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| media-content-production | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 |
| finance-economics | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **2 / 40** | **1 / 40** | **5 / 40** | **2 / 40** | **4 / 40** | **3 / 40** | **2 / 40** | **3 / 40** |

The account composite stays at **7/40**. Software 1 is this request's first software solve, but it equals the
software 1 already banked by v4, v6 and v8, so no domain beat its best.

## Predictions

- **R1, media: held.** One media task was solved. The request has now solved media in 5 of our 8 draws.
- **R2, new point: no.** The null rate was 0.08.
- **R3, total 2–5: held,** at 3.

## Redraws of the same request by other accounts

Four more redraws were scored on 11-03 (NoeFlandre's of 10-26 is in `result-v7r6.md`):

| Account | Run | Dataset | Total | Solved |
|---|---|---|---|---|
| kshitijthakkar (credited, #422) | `79fc6eae` | `29b93ada` | 5 | industrial 2, office 2, software 1 |
| wenbopan | `9756c6a6` | `29b93ada` | 4 | industrial 2, media 1, office 1 |
| Syndicate604 | `c27074b1` | own copy, `2cfd74ab` | 1 | office 1 |
| Syndicate604 | `59295eae` | own copy, `2cfd74ab` | 1 | office 1 |

## The v7 request after thirteen draws

- **Totals:** 2, 1, 5, 2, 4, 3, 2, 3 on our account (mean 2.75); 2, 5, 4, 1, 1 on others' (mean 2.6). The field's
  mean is 2.85.
- **Media:** solved in 6 of 13 draws, never more than one task.
- **Science and math:** trained in every draw, never solved.
- **New points:** none in our 8 draws. Under the field's rate of 0.08 per run, eight misses have P ≈ 0.51.

## Training, description only

Trackio was not queried for this run. Nothing is reported.

## Board context, 2026-11-04 at 14:30Z

270 `heldout-v2` runs have been evaluated. The per-run total is 2.85 ± 1.12, and no run has scored more than 5. A
run of another account produced a result that would be new for us in 20 of 254 runs (0.079).

Thirty runs by other accounts were scored after the r6 analysis (2026-10-29 07:50Z). Two would have been new for
us: rca-umb's Five Desks v21 (math 1; 4/40) and megasoft78's Strim test-fix (office 3; 4/40).

## v9

v9 (`profa-qe-arena-v9`, run `b302d037`, accepted 2026-10-29 10:11Z) is still `validated` with no evaluation
entry, six days later. This run was accepted a day after v9 and has been scored, so the stall is not the
arena's queue. v9's content is being redrawn under a new id (`predictions-v9r.md`).
