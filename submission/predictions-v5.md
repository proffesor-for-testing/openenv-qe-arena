# profa-qe-arena-v5: predictions written before submission

Entry: "QE terminal". The same QE graders sit behind a tbench2-shaped terminal interface (`exec` / `write_file` /
`evaluate`) with a real, uid-isolated shell. Budgets are 4096 completion / 8192 context tokens and 32 calls per
episode. There are 8 train cells, one per arena domain:

| Domain | Cell | Strict untrained solve rate (16 episodes) |
|---|---|---|
| industrial | qe_mutant_kill-industrial-d2 | 0.50 |
| office | qe_bugfix-office-d3 | 0.69 |
| finance | qe_mutant_kill-finance-d2 | 0.69 |
| security | qe_mutant_kill-security-d2 | 0.62 |
| science | qe_mutant_kill-science-d3 | 0.31 |
| media | qe_mutant_kill-media-d3 | 0.38 |
| math | qe_bugfix-math-d3 | 0.44 |
| software | qe_mutant_kill-d2 | 0.69 |

The solve rates come from the repaired `calibrate.py --interface terminal --strict`, run in the image as root on
ruos with Qwen/Qwen3.8-27B (thinking off). The image and dataset revision are recorded with the receipt.

## What calibration showed (read this before the predictions)
- **Per-episode compliance is already high.** Across 272 strict episodes, 11% ended on a reply that was not
  exactly one JSON object, and the strict mean was only 0.04 below the lenient mean. Under this interface,
  protocol failure is not the main loss. Capability is: mutant-kill expected values, and bug-fix at d3.
- **Budget ends are real at d3:** 19 of 144 re-screen episodes hit the 4096-token completion budget. The
  worst cell was qe_mutant_kill-finance-d3 at 7 of 12, and it was left out.
- So v5 tests "train the eval-shaped surface on solvable QE work". It does **not** test the stronger "train
  protocol compliance to saturation" hypothesis, because there is little compliance left to gain.

## Null model
Per-task solve rates are fitted to 78 leaderboard runs (plan, 2026-10-15). The total is 2–4 with
P(≥ 5) = 0.074 and P(≤ 1) = 0.089. Each domain's chance of at least one solve: office about 0.98,
industrial about 0.78, software about 0.30, finance about 0.10, media about 0.04, and math, security and
science about 0.01 each. Since #229 (an identical null control scored 1 then 0), the eval itself is not
deterministic. #232 reports an office drop from 2026-10-14.

## Predictions
- **P1, total:** 2–4 / 40, the null band.
- **P2, evidence for the method:** total ≥ 6 (null P = 0.010), **or** solves in ≥ 2 of software, finance,
  media, science, math and security (null P = 0.048). This must hold in v5 **and** in an identical replicate.
  One run reads as inconclusive.
- **P3, harmful transfer:** 0/40 (null about 0.5%). That would mean the terminal habit hurt the eval
  (cf. #223).
- **P4, banked composite (5/40 now: software 1, industrial 1, office 2, finance 1):** P(≥ 6) is about
  0.37 under the null, mostly from industrial reaching 2.
- **P5, training (Trackio, per task):**
  - mean reward rises from the first quarter to the last;
  - the share of steps with nonzero gradient is ≥ 40%, since every cell starts between 0.31 and 0.69;
  - no cell falls by more than 0.2 within the run.

## Reading rule
- A score inside P1 means no method claim, as with every run on the board.
- A single run satisfying P2 is inconclusive. Only a replicated P2 counts.
- If P5 fails (training did not learn), the score says nothing about the method either way.
