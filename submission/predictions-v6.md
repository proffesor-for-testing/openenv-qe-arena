# profa-qe-arena-v6: predictions written before submission

Entry: "light touch". This uses the same terminal image, dataset revision (`95b64de`), schema and per-task limits as v5, v5r
and v5r2. Only the 8 train cells change. They are now cells the untrained model already mostly solves, so most
GRPO groups have identical rewards and give no gradient:

| Domain | Cell | Strict untrained solves |
|---|---|---|
| software | qe_bugfix-d2 | 4/4 |
| industrial | qe_bugfix-industrial-d2 | 4/4 |
| finance | qe_bugfix-finance-d2 | 4/4 |
| security | qe_bugfix-security-d2 | 4/4 |
| office | qe_bugfix-office-d2 | 15/16 |
| science | qe_mutant_kill-science-d2 | 15/16 |
| media | qe_mutant_kill-media-d2 | 4/4 |
| math | qe_mutant_kill-math-d3 | 4/4 |

The solves come from the same strict terminal calibration as v5 (ruos `~/projects/v5-cal/`, Qwen/Qwen3.8-27B,
thinking off). Six cells have only 4 episodes, so their true rates could be well below 1.

## Why
- **#232 (dnakhla, 82 runs):** the only training fact that lines up with the private score is whether any gradient
  happened. Zero-gradient runs average 1.6, the rest 2.95. Runs with 1–25 gradient steps average 3.33 and runs with
  150+ average 2.40. dnakhla reads that second split as "the amount does not matter". It is observational and
  small-n, and easy environments may cause both few gradient steps and good scores.
- **Our three replicates (3, 3, 2):** v5r2 had the best training curve and the lowest score (`result-v5r2.md`).
  More training on the v5 cells has not helped.
- **#275 and #276 (NoeFlandre):** the same light-touch design on another environment. Its result is not known when
  this is written.
- The request has no training-step setting, so the task choice is the only way to get few gradient steps.

## Null model
The board's same-request spread is about 1 point (#259, #265, #280). Across all runs the mean is 2.77 with sd 1.1
(#275). Banked composite: 6/40 (software 1, finance 1, industrial 2, office 2). It rises only if industrial or
office reaches 3, or if cybersecurity, math, media or science gets a solve.

## Predictions
- **P1, the lever is applied (Trackio):** mean `frac_reward_zero_std` ≥ 0.70, so at most 30% of steps have a
  gradient (v5 runs: 54–60%). First-quarter mean reward ≥ 0.80 (v5 runs: 0.45–0.48).
- **P2, total:** 2–4 / 40. That is the null band, and P1 holding does not change it.
- **P3, evidence for the lever:** total ≥ 5. A single run is inconclusive, because the spread alone gives this about
  10% of the time.
- **P4, harmful:** 0–1 / 40, the level of the zero-gradient runs in #232. That would mean light touch behaves like no
  training.
- **P5, composite:** most likely stays at 6/40. P(≥ 7) is about 0.1. That figure comes from the v5 null model's
  per-domain chances (media about 0.04; math, security and science about 0.01 each) plus a small chance of a
  third industrial or office solve. Office has scored lower board-wide since 10-14 (#232).

## Reading rule
- If P1 fails (gradient share > 40%), calibration did not transfer to the arena's sampling. The run then does not
  test the lever, whatever it scores.
- If P1 holds, the score is one draw for the light-touch bucket next to our three v5 draws (3, 3, 2). No method claim
  from one run.
