# profa-qe-arena-v7r: result, read against predictions-v7.md

Draw 2 of 3 of the content-cell entry. Same request body as v7 apart from the id and name (request_sha256
`cb5e4b91…`), so the same seven `qe_skill` cells, image `terminal@sha256:9307e4e6…` and dataset revision `f5e0b95`.
Predictions are the ones committed for v7 in `36fd8a7`. The run was accepted on 2026-10-22 at 08:48:42Z as run
`1c6db4c10fe1d0559ddf934c`, and evaluated at 12:31Z after 190 steps on `heldout-v2`.

| Domain | v5 | v5r | v5r2 | v6 | v7 | v7r |
|---|---|---|---|---|---|---|
| industrial-physical-systems | 2 | 1 | 1 | 1 | 0 | 0 |
| office-white-collar | 1 | 2 | 1 | 1 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 1 | 0 | 0 |
| media-content-production | 0 | 0 | 0 | 0 | 1 | 0 |
| cybersecurity, finance-economics, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** | **2 / 40** | **3 / 40** | **2 / 40** | **1 / 40** |

## Predictions
- **P1, training signal:** not read. Trackio's metric API still times out, as it has since 2026-10-20 at about 11:54Z.
- **P2, total 2–4:** **missed** at 1.
- **P3, a science, math or media solve:** **missed.**
- **P4, composite ≥ 7 if and only if P3 holds:** this run adds no new best. The composite stays at 7/40 from v7's
  media solve.
- **P5, harm (0–1):** **in the band.** 1/40 is the level of zero-update runs.

## Reading
- This is our lowest score in nine runs. It is one draw. 19 of 171 evaluated board runs score 0 or 1 (about 0.11),
  and our own unchanged v5 body ranged 2 to 3 over three draws, so 1 is not far outside the noise. A second
  harm-band draw would be.
- The run lost v7's media solve and kept the office solve. Industrial stayed at 0, as in v7. Six of our seven
  code-QE runs solved 1 or 2 industrial tasks (v4 solved none); both content-cell runs solved none. That pattern is
  suggestive, not established.
- The null for a science, math or media solve is 13 of 171 evaluated board runs, about 0.076 per run.
- **Three-draw rule:** 1 of 2 draws so far has such a solve. Draw 3 decides it. A solve makes 2 of 3, which the
  pre-registered rule counts as evidence (P ≈ 0.014 under the null). No solve leaves 1 of 3, which is what the
  null produces.
- **Draw 3** (`profa-qe-arena-v7r2`, same body apart from the id and name) is built and dry-run with
  request_sha256 `44b5275a73a52a5108b61d3b7bc865bb34b4642c5b87b002d696f5f4058419a2` (file sha256 `3249131f…`). It
  waits for approval of that sha. The next slot opens about 2026-10-23 at 08:48:43Z.
