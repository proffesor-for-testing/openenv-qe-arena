# profa-qe-arena-v5r2: result, read against predictions-v5.md

The second replicate of v5. `request-v5r2.json` differs from `request-v5.json` and `request-v5r.json` only in
`submission_id` and `name` (request_sha256 `05b24b35…`). It was accepted on 2026-10-18 at 07:24Z as run
`6fb817d2648bd1b24bff42de` and evaluated at 11:05Z after 173 steps. Benchmark `heldout-v2`, same image and dataset
revision as v5.

| Domain | v5 | v5r | v5r2 |
|---|---|---|---|
| industrial-physical-systems | 2 | 1 | 1 |
| office-white-collar | 1 | 2 | 1 |
| cybersecurity, finance-economics, mathematics-or-formal-reasoning, media-content-production, natural-science, software-engineering | 0 each | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** | **2 / 40** |

## Predictions
- **P1, total 2–4:** held at 2, the bottom of the null band.
- **P2, method evidence:** not met in any of the three runs, so no method claim.
- **P3, harmful transfer (0/40):** not seen.
- **P4, banked composite:** stays at 6/40. v5r2 did not raise any domain's best.
- **P5, training signal (Trackio):** held, and it is the strongest of the three runs.
  - Mean reward by quarter was 0.45, 0.54, 0.61 and 0.71 (v5: 0.48 to 0.65, v5r: 0.45 to 0.62).
  - `frac_reward_zero_std` averaged 0.46, so 54% of steps had a nonzero gradient.
  - No cell fell. Security was flat.

    | Cell | First half | Second half |
    |---|---|---|
    | industrial | 0.41 | 0.73 |
    | office | 0.59 | 0.75 |
    | finance | 0.48 | 0.80 |
    | security | 0.57 | 0.57 |
    | science | 0.45 | 0.52 |
    | media | 0.30 | 0.40 |
    | math | 0.42 | 0.57 |
    | software | 0.75 | 0.93 |

## Reading
- **Three draws of one request: 3, 3 and 2.** Industrial went 2, 1, 1 and office 1, 2, 1. The spread is about one
  task per domain and one point in total, the same size rca-umb measured for five-desks (#259: 5 then 3).
- **The best training curve gave the lowest score.** v5r2's last-quarter reward (0.71) is the highest of the three, and
  its private score is the lowest. Within this request, the training signal does not predict the private score.
- **`completions/clipped_ratio` reads as in v5r.** Mean 0.86, rising from 0.72 to 0.94 by quarter. It is at least the
  step's reward on 164 of 173 steps, and `completions/max_length` reached 3000 tokens on 2 steps and never 4096. See
  `result-v5r.md`, "Reading"; the `mask_truncated_completions` question (#256) is still unanswered.
- **Consequence:** another unchanged replicate adds little. The three runs already bound the same-request spread, and
  the composite can only rise when a draw lands above a banked domain best.
