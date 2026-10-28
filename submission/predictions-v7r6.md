# profa-qe-arena-v7r6: predictions written before submission

Entry: v7 draw 7 (our 7th; NoeFlandre's redraw #381 makes 8 draws of this request in all). Dragan chose it on
2026-10-28 for the 09:21:32Z slot ("use v7 for today's slot"). The request is `request-v7r5.json` with only
`submission_id` and `name` changed: the same seven `qe_skill` content cells (science d2 and d3; math d1, d2 and d3;
media d1 and d2), the same image `terminal@sha256:9307e4e6…` and the same limits.

- request_sha256 (canonical): `0e373af7443ea360382b4b91e511b3eeb179d6ce254469fa621fd90edb6592fe`
- `request-v7r6.json` file sha256: `996d62f0…`
- The r5 request recomputes to `e8eea61a…` with the same digest function.
- Dry run passed on 2026-10-28 at about 09:00Z. Approved by Dragan ("arm it, I approve"); submitted 09:30:32Z, accepted 09:30:33Z (the timer was set after the slot opened and never fired, so the script was run directly).
- Dataset: HEAD `99808a44`, as in r4 and r5. The submit script does not submit if HEAD has moved.

## Why this request

It is a filler draw while v9 is being built. No other request we have can do better without new code. It adds a
composite point only through media ≥ 2, natural-science ≥ 1, math ≥ 1, or a domain it does not train (office ≥ 3,
industrial ≥ 3, finance ≥ 2, software ≥ 2, cybersecurity ≥ 1).

## Null model

From the 2026-10-27 research (`result-v7r5.md`): 17 of 215 other-account runs had a result that would be new for
us (0.079). Totals per run average 2.86 ± 1.11. Of the 7 runs by others since r5, one would have been new for us:
jjmachan's pool-desk-v7r (office 3).

## Predictions

- **R1, media:** at least one media task solved. About 0.57 on this request's record (media in 4 of 7 draws).
- **R2, new point:** P ≈ 0.08. In 7 draws this request has produced none, so we do not expect one.
- **R3, total:** 2–5, the range of this request.

There is **no training-health prediction.** Trackio metrics are reported as description only.

## Reading rule

- A new point is the only result that changes the composite.
- Whatever the outcome, this draw does not change the plan: v9 is the next request with new content.
