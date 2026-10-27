# profa-qe-arena-v7r5: result, read against predictions-v7r5.md

Draw 6 of the v7 request. The request body is v7's apart from the id and name (request_sha256 `e8eea61a…`). The
predictions were committed in `13a29f8` before submission.

The persistent submit timer on ruos fired at 09:21:30Z. The run was accepted at 09:21:31Z as run
`3279d7039bfd4411e0bc6e31`, and evaluated at 13:47Z on `heldout-v2` after 175 steps. The leaderboard records dataset
revision `99808a44`, so the dataset guard in `submit-v7r5.sh` held. The timer units were removed after the run was
accepted.

| Domain | v7 | v7r | v7r2 | v7r3 | v7r4 | v7r5 | NoeFlandre redraw |
|---|---|---|---|---|---|---|---|
| industrial-physical-systems | 0 | 0 | 2 | 0 | 1 | 0 | 1 |
| office-white-collar | 1 | 1 | 1 | 1 | 2 | 2 | 1 |
| software-engineering | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| media-content-production | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| finance-economics | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **2 / 40** | **1 / 40** | **5 / 40** | **2 / 40** | **4 / 40** | **3 / 40** | **2 / 40** |

The account composite stays at **7/40**. Office and finance matched their banked 2 and 1; no domain beat its best.

The last column is NoeFlandre's independent redraw of the public v7 request from their own account (board #381,
run `ba536227`, 167 steps, image republished with identical layers). It is shown because it is the same request,
not one of our draws.

## Predictions

- **R1, media: missed.** No media task was solved. Under the reading rule this does not reopen the series; it is
  the sixth draw of the request.
- **R2, new point: no.** The null rate was 0.08.
- **R3, total 2–5: held,** at 3.

## The v7 request after seven draws

- **Media:** solved in 4 of 7 draws (4 of our 6, 0 in NoeFlandre's), against 4 of 214 runs of other requests on
  other accounts (0.019). The two pre-registered tests (r3, r4) both held. The lift is real but has never exceeded
  one media task, and media 1 was banked by the first draw.
- **New points:** none in 7 draws. Under the field's rate of 0.079 per run, seven misses have P ≈ 0.56, so the
  request is not shown to be worse than the field, but nothing shows it is better for the composite either.
- **Science and math:** trained in every draw, never solved.
- **Totals:** 2, 1, 5, 2, 4, 3 (mean 2.83), and 2 on NoeFlandre's account; the field's mean is 2.86.

## Training, description only

Trackio logged 175 points. Mean reward by quarter was 0.84, 0.83, 0.85 and 0.90. The share of groups with zero
reward variance rose from 0.53 to 0.70 by quarter, and 68 of 175 steps had a non-zero gradient norm.

## Board context, 2026-10-27 at 16:00Z

229 runs have been evaluated. The per-run total is 2.86 ± 1.11, and no run has scored more than 5.

A run of another account produced a result that would be new for us (science ≥ 1, math ≥ 1, cybersecurity ≥ 1,
office ≥ 3, industrial ≥ 3, or media, finance or software ≥ 2) in 17 of 215 runs (0.079). Grouped by account and
environment family (families with 3 or more runs), only one family is above that rate:

| Account / family | Runs | New-for-us results | Binomial P under 0.079 |
|---|---|---|---|
| jjmachan / pool-desk (v1, v2, v5, v6) | 11 | 5 (industrial 3, science 1, office 3 twice, cybersecurity 1) | 0.001 |
| kshitijthakkar / terminal-spr-lab | 3 | 1 | 0.22 |
| ruv / metaharness-arena-tasks | 12 | 2 | 0.24 |
| dnakhla / terminal-desk | 13 | 2 | 0.28 |
| 18 other families on other accounts | 3–16 each | 0–1 | ≥ 0.34 |

pool-desk-v6 has the highest mean of any dataset with 3 or more runs (4.0 over 5 runs). The family grouping was
chosen after looking at the data, so the P value overstates the evidence somewhat. Even so, a Bonferroni
correction over the 22 families still gives P ≈ 0.02.

Two media replications by others missed: NoeFlandre's redraw of the v7 request (above), and kshitijthakkar's
terminal-spr run that raised media tasks from 7 to 21 of 50 with a different subtitle family (board #385, media 0).

## Next

The next slot opens about 2026-10-28 at 09:21:32Z. Nothing is armed. The choice of request is open.
