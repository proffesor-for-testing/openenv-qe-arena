# profa-qe-arena-v4: predictions written before submission

Entry: QE across domains, 11 train tasks.
- Mutant-kill on domain code: industrial, office, finance, security and science, all at d2.
- Bug-fix: math d2.
- Flaky: media d2 and industrial d2.
- Software: mutant-kill d3, bug-fix d4 and flaky d2.

Image and dataset revision are recorded with the receipt.

## Null model
Base rates come from the public leaderboard (59 evaluated runs, analysed 2026-10-13):
- no predictor of the score (train-task count p=0.43, steps p=0.31, user p=0.68);
- per-domain "at least one solve" rates: office 0.98, industrial 0.78, software 0.30, finance 0.06, media 0.04, math 0.02, security 0.02, science 0.00;
- the rca-umb collapse control (#201) and #191 show no within-run training effect on the private score.

## Predictions
- **P1, total:** 2–5 / 40. Null: about 90% of runs fall here.
- **P2, industrial = 2:** about 30%. Software ≥ 1: about 30%. Office ≥ 1: about 98%.
- **P3, domain targeting does not show:**
  - finance ≥ 1 about 6%, security ≥ 1 about 2%, science ≥ 1 about 0%, media ≥ 1 about 4%, math ≥ 1 about 2%;
  - we expect these at their base rates. A solve in science (never seen in 59 runs) or a second domain off base rate would be notable but not conclusive (n = 1).
- **P4, banked composite (currently 4/40: finance 1, industrial 1, office 2):** P(≥ 5) about 0.5–0.6, mostly from industrial 2 or software 1.
- **P5, training signal (Trackio):**
  - mean reward rises from the first quarter to the last;
  - the share of steps with nonzero gradient is ≥ 40% (v2: 37–45%);
  - mutant-kill cells do not fall within the run, unlike v2's d1 (0.94→0.74).

  If mutant-kill falls again, we read that as family interference and shrink the mutant-kill share in v5.

## Reading rule
- A score inside P1 with P3 at base rates means no method claim, as with every run on the board.
- Only a replicated off-base-rate domain result would count as evidence that domain-matched QE training moves the private score.
