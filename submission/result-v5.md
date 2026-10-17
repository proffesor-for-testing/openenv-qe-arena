# profa-qe-arena-v5: result, read against predictions-v5.md

Run `e488b58106cd6dc1192f6758` was validated on 2026-10-15 at 18:08Z and evaluated 166 steps later.
Benchmark `heldout-v2`, dataset revision `95b64de`.

| Domain | Solved / 5 |
|---|---|
| industrial-physical-systems | 2 |
| office-white-collar | 1 |
| cybersecurity, finance-economics, mathematics-or-formal-reasoning, media-content-production, natural-science, software-engineering | 0 each |
| **Total** | **3 / 40** |

## Predictions
- **P1, total 2–4:** held at 3. This is the null band, so no method claim.
- **P2, method evidence:** not met. The total is below 6 and no rare domain was solved. The replicate is still owed by the
  reading rule, but P2 can now only come from a combined reading, and v5 alone contributed nothing to it.
- **P3, harmful transfer (0/40):** not seen.
- **P4, banked composite ≥ 6:** held. Industrial reached 2, so the composite went from 5/40 to 6/40 (average 0.15).
- **P5, training signal (Trackio):** held.
  - Mean reward by quarter was 0.48, 0.48, 0.56 and 0.65.
  - `frac_reward_zero_std` averaged 0.40, so 60% of steps had a nonzero gradient.
  - No cell fell. These are first-half against second-half reward per cell. We assume `step/task` *i* is the *i*-th entry
    in tasks-v5.json; Trackio does not label it.

    | Cell | First half | Second half |
    |---|---|---|
    | industrial | 0.45 | 0.75 |
    | office | 0.62 | 0.66 |
    | finance | 0.38 | 0.66 |
    | security | 0.62 | 0.75 |
    | science | 0.38 | 0.48 |
    | media | 0.20 | 0.23 |
    | math | 0.35 | 0.53 |
    | software | 0.78 | 0.85 |

## Reading
Training learned the train cells, and the private score stayed at the null. This matches every earlier run on the
board: a within-run training signal has not been seen to move the score. One observation for the next design:
`completions/clipped_ratio` averaged 0.84 and rose from 0.76 to 0.93 across the run. Most completions reached the
length cap, and more did as training went on. Whether that counts whole multi-turn episodes or single turns is not
documented, so we do not read it as a defect yet.

**Correction (2026-10-17):** "most completions reached the length cap" is wrong. In TRL the metric counts
completions whose last token is not EOS, and at least 98% of solved completions count as clipped. See
`result-v5r.md`, "Reading".
