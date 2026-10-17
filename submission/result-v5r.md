# profa-qe-arena-v5r: result, read against predictions-v5.md

The replicate of v5. `request-v5r.json` differs from `request-v5.json` only in `submission_id` and `name`
(request_sha256 `2c3744a4…`). It was accepted on 2026-10-16 at 18:08Z as run `f4fa9d64b6a68f9a8f388eaf` and evaluated
at 21:51Z after 167 steps. Benchmark `heldout-v2`, same image and dataset revision as v5.

| Domain | v5 | v5r |
|---|---|---|
| industrial-physical-systems | 2 | 1 |
| office-white-collar | 1 | 2 |
| cybersecurity, finance-economics, mathematics-or-formal-reasoning, media-content-production, natural-science, software-engineering | 0 each | 0 each |
| **Total** | **3 / 40** | **3 / 40** |

## Predictions
- **P1, total 2–4:** held at 3. The null band again.
- **P2, method evidence:** not met in v5r, and not met in v5. The reading rule needs P2 in both runs, so the pair gives
  no evidence for the method.
- **P3, harmful transfer (0/40):** not seen.
- **P4, banked composite:** stays at 6/40. v5r did not raise any domain's best.
- **P5, training signal (Trackio):** held.
  - Mean reward by quarter was 0.45, 0.50, 0.51 and 0.63.
  - `frac_reward_zero_std` averaged 0.44, so 56% of steps had a nonzero gradient.
  - No cell fell by more than 0.2. Two cells fell slightly: security 0.55 to 0.45 and media 0.35 to 0.30.

    | Cell | First half | Second half |
    |---|---|---|
    | industrial | 0.41 | 0.57 |
    | office | 0.59 | 0.78 |
    | finance | 0.41 | 0.70 |
    | security | 0.55 | 0.45 |
    | science | 0.35 | 0.43 |
    | media | 0.35 | 0.30 |
    | math | 0.38 | 0.52 |
    | software | 0.75 | 0.82 |

## Reading
- **Same request, same training curve, different domains.** The two runs swapped industrial and office solves at the
  same total. That is the nondeterminism #229 reported, and it is the main thing the replicate measured.
- **`completions/clipped_ratio` is not length truncation here.** It averaged 0.85 (v5: 0.84). TRL 1.16 computes it as
  "the completion's last token is not EOS or pad" (`grpo_trainer.py`, around line 2302); it does not compare the
  length with the cap. In both runs it is at least the step's reward on 161 of 167 steps (v5: 162 of 166). So at least
  98% of solved completions count as "clipped", and the 16% that terminate with EOS contain at most 6 solves.
  Real length pressure is rare: `completions/max_length` reached 3000 tokens on 3 of 167 steps (v5: 1) and never
  4096. The likely mechanism is that the arena's rollout ends the completion on the final environment observation
  after `evaluate`, so the last token is observation text. A rising ratio then means more episodes end through
  `evaluate`, which is consistent with the rising reward. This is an inference from public metrics and TRL source;
  the arena's trainer code and settings are not public.
- **One open risk follows from it.** If the arena sets TRL's `mask_truncated_completions=True` (TRL's default is
  False), the "clipped" completions, including nearly every solved one, would be dropped from the loss. Public
  metrics cannot tell the two settings apart. That is a question for the operators (compare #226).
