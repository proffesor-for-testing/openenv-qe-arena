# profa-qe-arena-v7r7: predictions written before submission

Entry: v7 draw 8 on our account (9 draws of this request in all, counting NoeFlandre's redraw #381; kshitijthakkar's
credited redraw #422 is also in flight). Dragan chose it on 2026-10-30 for the slot that opens about 10:11:09Z
("Submit another v7 draw"). The request is `request-v7r6.json` with only `submission_id` and `name` changed: the
same seven `qe_skill` content cells (science d2 and d3; math d1, d2 and d3; media d1 and d2), the same image
`terminal@sha256:9307e4e6…` and the same limits.

- request_sha256 (canonical): `091a8d870947e4d398c34afb56f7dd8bf3cbac3fa5893d11164c3f92ab345f67`
- Dataset: HEAD `29b93ada`, not `99808a44` as in r4–r6. The request does not pin a revision, and HEAD moved for v9.
  The 64 rows of each of the seven v7 cells are byte-identical in both revisions (checked 2026-10-30), so the
  training data is unchanged. The submit script does not submit if HEAD has moved again.
- v9 (run `b302d037`, accepted 2026-10-29 10:11Z) is still `validated` with no evaluation. Runs other accounts
  submitted after it were scored on 2026-10-30 between 02:26 and 02:36Z.

## Why this request

v9 has no result to repeat or extend, and no other request is ready. This draw adds a composite point only through
media ≥ 2, natural-science ≥ 1, math ≥ 1, or a domain it does not train (office ≥ 3, industrial ≥ 3,
finance ≥ 2, software ≥ 2, cybersecurity ≥ 1).

## Null model

From the leaderboard on 2026-10-30 at 09:39Z (249 `heldout-v2` runs, 234 by other accounts):

- A result that would be new for us: 19 of 234 (0.081).
- Totals: 2.85 ± 1.11, none above 5.

## Predictions

- **R1, media:** at least one media task solved. About 0.5 on this request's record (media in 4 of 8 draws).
- **R2, new point:** P ≈ 0.08. Eight draws of this request have produced none, so we do not expect one.
- **R3, total:** 2–5, the range of this request (1–5 across its draws).

There is **no training-health prediction.** Trackio metrics are reported as description only.

## Reading rule

- A new point is the only result that changes the composite.
- Whatever the outcome, this draw does not change the plan. v9's three-draw series resumes once draw 1 is scored or
  its stall is explained.
