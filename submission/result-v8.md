# profa-qe-arena-v8: result, read against predictions-v8.md

v8 trains five `qe_skill` content cells: science d2 and d3, and media d1, d2 and d3 (request_sha256 `535b668b…`).
The predictions were committed in `a74c153` before submission.

The run was accepted on 2026-10-25 at 08:51:32Z as run `282770faa3c8c40765f99165`. It was evaluated at 12:59Z on
`heldout-v2` after 155 steps. The leaderboard records dataset revision `99808a44`, the upload with the media-d3
rows, so the dataset guard in `submit-v8.sh` held.

| Domain | v7 | v7r | v7r2 | v7r3 | v8 | Banked best |
|---|---|---|---|---|---|---|
| industrial-physical-systems | 0 | 0 | 2 | 0 | 0 | 2 |
| office-white-collar | 1 | 1 | 1 | 1 | 1 | 2 |
| software-engineering | 0 | 0 | 0 | 0 | 1 | 1 |
| media-content-production | 1 | 0 | 1 | 1 | 0 | 1 |
| finance-economics | 0 | 0 | 1 | 0 | 0 | 1 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **2 / 40** | **1 / 40** | **5 / 40** | **2 / 40** | **2 / 40** | **7 / 40** |

The account composite stays at **7/40**. No domain beat its banked best.

## Predictions

- **V1, media: missed.** No media task was solved. The reading rule rates a miss as weak evidence against
  media-weighted training (P ≈ 0.33–0.5 at the hypothesised rate).
- **V2, media ≥ 2: missed.** No new point.
- **V3, total 2–4: held,** at 2.
- **V4, no loss elsewhere: held.** Every domain is within 0–2. Software-engineering scored 1, its third solve in
  12 runs (after v4 and v6); the content cells do not train it. Office scored in all 12.

## Training, description only

Trackio logged 155 points. Mean reward by quarter was 0.71, 0.67, 0.81 and 0.84, lower than v7r3's 0.86 → 0.89.
That fits media d3 being the harder cell, and there was no training-health prediction.

## Board context, 2026-10-25 at 15:00Z

207 runs have been evaluated. The per-run total is 2.86 ± 1.12. Other accounts solved a media task in 4 of 195 runs
(0.021), and no run on any account has solved two.

## Next

Per the v8 reading rule, the choice between v8 and v7 waits for the remaining r-series draws. The next slot (about
2026-10-26 at 08:51:32Z) goes to r4, the v7 request unchanged apart from id and name (`predictions-v7r4.md`).
