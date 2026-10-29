# profa-qe-arena-v7r6: result, read against predictions-v7r6.md

Draw 7 of the v7 request (8 in all, counting NoeFlandre's redraw). The request body is v7's apart from the id and
name (request_sha256 `0e373af7…`). The predictions were committed in `67b8e6f` with the receipt, written before the
submission was approved.

The run was submitted by hand at 09:30:32Z on 2026-10-28 and accepted at 09:30:33Z as run
`119ff56294701c6c0287552a`. It was evaluated at 21:01Z on `heldout-v2` after 182 steps. The leaderboard records
dataset revision `99808a44`, so the dataset guard held.

The arena scored nothing for any account between 09:30Z and 21:01Z on 2026-10-28. Our run and two others were
scored within six minutes of each other when it resumed, so the delay was the arena's queue, not this request.

| Domain | v7 | v7r | v7r2 | v7r3 | v7r4 | v7r5 | v7r6 | NoeFlandre redraw |
|---|---|---|---|---|---|---|---|---|
| industrial-physical-systems | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 1 |
| office-white-collar | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| media-content-production | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| finance-economics | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **2 / 40** | **1 / 40** | **5 / 40** | **2 / 40** | **4 / 40** | **3 / 40** | **2 / 40** | **2 / 40** |

The account composite stays at **7/40**. No domain beat its best.

## Predictions

- **R1, media: missed.** No media task was solved. The request has now solved media in 4 of 8 draws (4 of our 7).
- **R2, new point: no.** The null rate was 0.08.
- **R3, total 2–5: held,** at 2.

## The v7 request after eight draws

- **Media:** solved in 4 of 8 draws, against 4 of 225 runs of other requests on other accounts. It has never
  solved more than one media task, and media 1 was banked by the first draw.
- **New points:** none in 8 draws. Under the field's rate of 0.08 per run, eight misses have P ≈ 0.51.
- **Science and math:** trained in every draw, never solved.
- **Totals:** 2, 1, 5, 2, 4, 3, 2 on our account (mean 2.71), and 2 on NoeFlandre's; the field's mean is 2.85.

## Training, description only

Trackio returned no data for this run (`get_metric_values` gave an empty list for reward, zero-std share and grad
norm when queried on 2026-10-29). Nothing is reported.

## Board context, 2026-10-29 at 07:50Z

240 `heldout-v2` runs have been evaluated. The per-run total is 2.85 ± 1.10, and no run has scored more than 5.
A run of another account produced a result that would be new for us in 18 of 225 runs (0.080).

Ten runs by other accounts were scored after the r5 analysis (2026-10-27 16:00Z). One would have been new for us:
jjmachan's pool-desk-v7 (office 3, industrial 2; 5/40). The others scored 1–3, all from industrial, office and
software.

## Next

v9 (desk tier: ghg, ledger and quakes at d4) was merged as `4187c74` on 2026-10-29. It is the next request with
new content if its image, dataset export and request are ready for the slot that opens about 2026-10-29 09:30:34Z.
If not, the fallback is another v7 draw.
