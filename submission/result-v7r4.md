# profa-qe-arena-v7r4: result, read against predictions-v7r4.md

Draw 5 of the v7 request, and draw 2 of the media series (r3 to r5). The request body is v7's apart from the id and
name (request_sha256 `9e282c52…`). The predictions were committed in `df21168` before submission.

The submit timer on ruos was lost when the machine restarted at about 08:02Z (a transient `systemd-run` timer does
not survive a reboot). The same approved script was run by hand at 09:20:46Z, 29 minutes after the slot opened. The
run was accepted at 09:20:47Z as run `9860c7f871e6442c9dcbf02a`, and evaluated at 14:24Z on `heldout-v2` after 181
steps. The leaderboard records dataset revision `99808a44`.

| Domain | v7 | v7r | v7r2 | v7r3 | v8 | v7r4 |
|---|---|---|---|---|---|---|
| industrial-physical-systems | 0 | 0 | 2 | 0 | 0 | 1 |
| office-white-collar | 1 | 1 | 1 | 1 | 1 | 2 |
| software-engineering | 0 | 0 | 0 | 0 | 1 | 0 |
| media-content-production | 1 | 0 | 1 | 1 | 0 | 1 |
| finance-economics | 0 | 0 | 1 | 0 | 0 | 0 |
| cybersecurity, mathematics-or-formal-reasoning, natural-science | 0 each | 0 each | 0 each | 0 each | 0 each | 0 each |
| **Total** | **2 / 40** | **1 / 40** | **5 / 40** | **2 / 40** | **2 / 40** | **4 / 40** |

The account composite stays at **7/40**. Office matched its banked 2; no domain beat its best.

## Predictions

- **M1, media (the test): held.** One media task was solved. Under the null, P = 0.021.
- **M2, new point: no.** The null rate was 0.07.
- **M3, total 2–4: held,** at 4.

## Media series: confirmed

The series is **2 of 2** (r3 and r4), which meets the pre-registered rule (2 or more of 3). Under the null, two
media solves in two draws has P ≈ 0.0004. r5 is not needed for the series.

What the result shows:
- Training the v7 content cells lifts the media solve rate. Media was solved in 4 of 5 draws of the v7 request,
  against 4 of 204 runs on other accounts (0.020).
- Only the last two of those draws were clean tests, since media was singled out after the first three.

What it does not show:
- Media is capped at one task per run so far. No run on the board has solved two, and v8, which aimed at two,
  solved none.
- Science and math were trained in all five draws and never solved.
- The lift has not moved the composite. Media was banked at 1 by the first v7 draw.

## Training, description only

Trackio logged 181 points. Mean reward by quarter was 0.85, 0.78, 0.87 and 0.92, close to v7r3's 0.86 → 0.89.

## Board context, 2026-10-26 at 14:30Z

217 runs have been evaluated. The per-run total is 2.86 ± 1.11, and 65 runs scored 4 or more. On other accounts'
runs, the chance of beating one of our per-domain bests is highest for natural-science ≥ 1 and office ≥ 3 (6 of 204
each), then math ≥ 1 and cybersecurity ≥ 1 (2 of 204 each). No run on any account has reached media ≥ 2, finance ≥ 2
or software ≥ 2.

## Next

The series is closed. The next slot opens about 2026-10-27 at 09:20:48Z.
