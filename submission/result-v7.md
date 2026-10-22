# profa-qe-arena-v7: result, read against predictions-v7.md

Draw 1 of 3 of the content-cell entry. Seven `qe_skill` cells: science d2 and d3, math d1 to d3, and media d1 and
d2. Image `terminal@sha256:9307e4e6…` (main `7490e2d`), dataset revision `f5e0b95`. Predictions were committed in
`36fd8a7` before submission (request_sha256 `6aef0b10…`). The run was accepted on 2026-10-21 at 08:11:05Z as run
`f3bb23ec709fca9d74342dd6`, and evaluated at 11:53Z after 187 steps on `heldout-v2`.

| Domain | v5 | v5r | v5r2 | v6 | v7 |
|---|---|---|---|---|---|
| industrial-physical-systems | 2 | 1 | 1 | 1 | 0 |
| office-white-collar | 1 | 2 | 1 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 1 | 0 |
| media-content-production | 0 | 0 | 0 | 0 | **1** |
| cybersecurity, finance-economics, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** | **2 / 40** | **3 / 40** | **2 / 40** |

## Predictions
- **P1, training signal:** not read. Trackio's metric API has timed out since 2026-10-20 at about 11:54Z, though the
  Space reports `RUNNING`. Until P1 is read, this run does not show that the cells trained.
- **P2, total 2–4:** held at 2.
- **P3, a science, math or media solve:** **held**, media 1/5. This is our first solve in any of those domains in 8
  runs.
- **P4, composite ≥ 7 if and only if P3 holds:** **held.** The composite went from 6 to 7/40 with the media best.
  Industrial dropped to 0 in this run, but banked bests cannot fall.
- **P5, harm (0–1):** not seen.

## Reading
- By the reading rule, one run is inconclusive. The P3 hit is a composite gain, not yet evidence for content training.
- The null for a science, math or media solve is now 13 of 169 evaluated board runs, about 0.077 per run, including
  this one. 0.07 still stands.
- The media solve is one task out of five, and we cannot tell which private task it was. A media solve in a run that
  trained on media cells is what the lever predicts. It is also what a 0.077 null produces about once in 13 runs.
- No solve in math or science, though five of the seven cells trained there.
- Comparison on the same day: Codcordance concordance-mixed-v1 trained on 4 original exact-math tasks and scored 4/40
  with no math solve. kshitijthakkar's all-domains terminal mix got a science solve (#329). nice-bill's shiftdesk-v11
  got a media solve.
- **Draw 2** (`profa-qe-arena-v7r`, same body apart from the id and name, request_sha256 `cb5e4b91…`) was accepted on
  2026-10-22 at 08:48:42Z and validated at 08:50Z as run `1c6db4c10fe1d0559ddf934c`. A third draw follows. By the pre-registered rule, 2 or more of 3 draws with such a solve is
  evidence (P ≈ 0.014 under the null).
