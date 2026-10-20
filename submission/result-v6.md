# profa-qe-arena-v6: result, read against predictions-v6.md

The light-touch run. It used the same image, dataset revision (`95b64de`), schema and limits as v5, but trained on 8
cells the untrained model mostly solves (`predictions-v6.md`, committed in `e04c364` before submission;
request_sha256 `55301b49…`). It was accepted on 2026-10-20 at 08:10Z as run `de42b49746bbf48e32f18654` and evaluated
at 11:54Z after 182 steps on `heldout-v2`.

| Domain | v5 | v5r | v5r2 | v6 |
|---|---|---|---|---|
| industrial-physical-systems | 2 | 1 | 1 | 1 |
| office-white-collar | 1 | 2 | 1 | 1 |
| software-engineering | 0 | 0 | 0 | 1 |
| cybersecurity, finance-economics, mathematics-or-formal-reasoning, media-content-production, natural-science | 0 each | 0 each | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** | **2 / 40** | **3 / 40** |

## Predictions
- **P1, the lever is applied:** not read yet. Trackio's metric API has not answered since this run was evaluated:
  every call returns nothing or times out with no connection. rca-umb reported the same outage in #303. This section
  will be updated once the metrics can be read. Until P1 is read, the run does not test the lever.
- **P2, total 2–4:** held at 3.
- **P3, evidence for the lever (total ≥ 5):** not met.
- **P4, harmful (0–1):** not seen. Light touch did not behave like no training.
- **P5, composite:** held at 6/40. The software solve matches the banked software 1 (v4).

## Reading
- Training on mostly-solved cells scored the same as training on the v5 cells (3, 3, 2), with no gain and no harm.
  That agrees with NoeFlandre's light-touch v3 (3/40, #276) and with rca-umb's #300, where the number of updating
  steps shows no correlation with the score across 131 runs.
- v6 is our first terminal run with a software solve. Its training set had one software cell (`qe_bugfix-d2`), as v5
  did (`qe_mutant_kill-d2`), so one draw does not separate the cell from noise.
