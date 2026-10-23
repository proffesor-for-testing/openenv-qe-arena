# profa-qe-arena-v7r2: result, read against predictions-v7.md

Draw 3 of 3 of the content-cell entry. It uses the same request body as v7 and v7r apart from the id and name
(request_sha256 `44b5275a…`): the same seven `qe_skill` cells, the same image `terminal@sha256:9307e4e6…` and the
same dataset revision `f5e0b95`. The predictions are the ones committed for v7 in `36fd8a7`.

The run was accepted on 2026-10-23 at 08:49:31Z as run `0afe55255c3d17ecf338320d`. It was evaluated at 12:34Z on
`heldout-v2` after 192 steps.

| Domain | v5 | v5r | v5r2 | v6 | v7 | v7r | v7r2 |
|---|---|---|---|---|---|---|---|
| industrial-physical-systems | 2 | 1 | 1 | 1 | 0 | 0 | 2 |
| office-white-collar | 1 | 2 | 1 | 1 | 1 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| media-content-production | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| finance-economics | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** | **2 / 40** | **3 / 40** | **2 / 40** | **1 / 40** | **5 / 40** |

This is our best single run in ten. The account composite stays at **7/40**, because no domain beat its banked
best.

## Predictions

### P1, training signal: holds, at the margin

Trackio logged the whole run, 192 points. This is the first content-cell run whose training signal we could read.
The v7 and v7r metrics were lost in the 10-19 to 10-21 outage (board #342, #343).

- **Mean reward rose.** It went from 0.818 in the first quarter to 0.911 in the last.
- **77 of 192 steps had a nonzero gradient.** That is 40.1%, against a threshold of 40%.
- **The cells saturated.** By quarter, the steps with a gradient were 25, 21, 18 and 13 of 48. The share of groups
  with zero reward variance rose from 0.48 to 0.73. Most such groups were all-solved.
- **Cause:** the cells were calibrated to 4–12 of 16 untrained strict solves, and the model learned them within a
  few visits. The run was healthy early and nearly idle late.
- **Clipped ratio:** mean 0.95, while the mean completion length was 495 tokens. This is the TRL "last token not
  EOS" flag already explained for v5, not truncation (rUv's metaharness PR #383 reached the same reading).

### P2 to P5

- **P2, total 2–4:** **missed high**, at 5. Before this run, 10 of 183 evaluated board runs had scored 5 or more.
- **P3, a science, math or media solve:** **held** (media 1).
- **P4, composite ≥ 7 if and only if P3 holds in a domain we have not solved:** **consistent.** Media had already
  been banked by v7, so the solve does not raise the composite. It stays at 7.
- **P5, harm (0–1):** **not seen.**

## Three-draw rule: met

Two of the three draws have a science, math or media solve: v7 (media 1) and v7r2 (media 1). The pre-registered
rule counts this as evidence for content training.

| Hypothesis | Rate per run | P(2 or more of 3) |
|---|---|---|
| Board null | 0.07 | ≈ 0.014 |
| Content-shell rate | 0.23 | ≈ 0.13 |

The likelihood ratio is about 9 in favour of the content-shell rate. Today's board null is 14 of 184 evaluated runs
(0.076), counting our two draws.

What the evidence does not show:
- **Media only.** Both solves are one media task. Five of the seven cells trained science or math, and neither
  domain was solved in any of the three draws. The evidence points at media, not at science and math.
- **The media-specific rate is post hoc.** Excluding our runs, 4 of 174 board runs solved a media task (0.023).
  Two of three at that rate is P ≈ 0.002, but media was singled out after the draws. Only new draws can test it
  cleanly.
- **The total of 5 is not the content cells.** The jump from 1 to 5 came from industrial 2 and finance 1, which are
  domains the cells do not train. Board-wide, totals do not correlate with step count (r = −0.04) or task count
  (r = −0.07) across 184 runs. A 5 is what a run lands on now and then.
- **Training health does not predict the score here.** The late-run saturation, with a zero-variance share of 0.73,
  coincided with our best score. Elsewhere on the board, update-step counts did not predict scores either (#300).

## Board context, 2026-10-23 12:47Z

| Account | Runs | Composite | Mean per run | Approach |
|---|---|---|---|---|
| ruv | 11 | 9 | 3.27 | Synthetic procedural tasks with partial-credit semantic reward. 8 "mixed-signal" cells, about 275 steps. An unattended daily runner re-draws the incumbent request. |
| jjmachan | 9 | 9 | 3.11 | A reviewed pool of real-data terminal tasks with pytest verifiers, 30–50 tasks per run, each at a 0.2–0.8 predicted solve rate. Pre-registered with a frozen selection rule. Banked domains are dropped from later runs. |
| Profa-QE | 10 | 7 | 2.90 | Code QE (7 runs), then content cells (3 draws) |

- **No account's mean beats the untrained reference.** The base model, before any training, scored 4/40 in one
  attempt per task. The board's per-run mean is 2.84 ± 1.11.
- **The composite rewards the number of draws.** It is a best-per-domain maximum over runs. We drew 10 random board
  runs at a time, 20,000 times. This null gives a median composite of 6, and P(composite ≥ 7) = 0.48. Our 7 is
  ordinary for 10 runs.
- **ruv and jjmachan are in the tail.** Their 9s have P ≈ 0.03 and 0.02 under this null. Across all accounts the
  null expects 0.42 accounts at 9 or more, and there are 2. That is unusual (P ≈ 0.07) but not decisive.
- **jjmachan states the same reading.** Their pre-registration says the eval re-samples on every run, so the
  objective is per-run P(new domain point), and each run is "another ticket".

## What a new point needs

Our banked bests are industrial 2, office 2, software 1, finance 1 and media 1. These are the routes to a new point:

| Route | Board rate per run |
|---|---|
| office ≥ 3 | 5 of 184 |
| natural-science ≥ 1 | 6 of 184 |
| mathematics ≥ 1 | 2 of 184 |
| cybersecurity ≥ 1 | 1 of 184 |
| industrial ≥ 3 | 1 of 184 |
| media, finance or software ≥ 2 | never seen |

Under the board null, a run adds a point about 8% of the time. If the media lift is real, media ≥ 2 becomes a live
route that the null does not have.
