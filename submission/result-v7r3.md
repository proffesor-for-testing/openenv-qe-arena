# profa-qe-arena-v7r3: result, read against predictions-v7r3.md

Draw 4 of the v7 request, and draw 1 of the media series (r3 to r5). The request body is v7's apart from the id and
name (request_sha256 `20226006…`). The predictions were committed in `db5eb1d` before submission.

The run was accepted on 2026-10-24 at 08:50:16Z as run `9feab29ca52ef6e313b75d08`. It was evaluated at 12:32Z on
`heldout-v2` after 182 steps. The leaderboard records dataset revision `f5e0b952`, the same as draws 1–3. The v8
upload waited for this evaluation and landed at 12:45Z (`99808a44`).

| Domain | v5 | v5r | v5r2 | v6 | v7 | v7r | v7r2 | v7r3 |
|---|---|---|---|---|---|---|---|---|
| industrial-physical-systems | 2 | 1 | 1 | 1 | 0 | 0 | 2 | 0 |
| office-white-collar | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| media-content-production | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| finance-economics | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** | **2 / 40** | **3 / 40** | **2 / 40** | **1 / 40** | **5 / 40** | **2 / 40** |

The account composite stays at **7/40**. No domain beat its banked best.

## Predictions

- **M1, media (the test): held.** One media task was solved. Under the null, P = 0.023.
- **M2, new point: missed.** Media stayed at 1, and no other domain beat its best. The null rate was 0.08.
- **M3, total 2–4: held,** at 2.

## Media series

With one draw of up to three made, the series stands at **1 of 1**.
- One more media solve, in r4 or r5, confirms the lift under the pre-registered rule. Two misses leave the result
  inconclusive (1 of 3).
- Across all four draws of the v7 request, media was solved in 3. Only r3 counts toward the series, since media was
  picked after draws 1–3 were seen.
- The base rate has barely moved. On today's board, 4 of 195 runs on other accounts solved a media task (0.021), and
  no run on any account has solved two.

What it does not show:
- Media is still the only content domain that moves. Science and math were trained in all four draws, and neither
  has been solved.
- The other point came from office, which the content cells do not train. Office has scored at least 1 in all 11 of our runs (2 in four of them).

## Training, description only

Trackio logged 182 points. Mean reward by quarter was 0.856, 0.81, 0.90 and 0.89. That is as high as v7r2's
0.82 → 0.91, and v7r2 had saturated cells. There was no training-health prediction.

## Board context, 2026-10-25

206 runs have been evaluated. The per-run total is 2.86 ± 1.12, unchanged from 2.84 ± 1.11 at 184 runs.

## Next

v8 (`535b668b`, run `282770fa`) was accepted on 2026-10-25 at 08:51:32Z and is training. It is read against
`predictions-v8.md`. The next slot opens about 2026-10-26 at 08:51:33Z. It goes either to r4 (the v7 request) or
to a second v8 draw, decided after v8's result.
