<!-- first committed in the project repository as 1baa94c at 2026-11-06T14:08:07+01:00 -->

## PORT-L amendment 1 (2026-11-06): B* and page score for Phase 3 (written after R4 and PORT-D D-A were read; before any L2/L3 data; D-B partially run, not read)

**B\* = 49152.** R4 passed at 49152 (Δ −1.91, CI [−4.46, 0.63]) and at 65536; 49152 is the smallest passing budget, so B\* = 49152 and 2B\* = 98304 by the PORT-L rule.

**Page score: quest:d2.** PORT-D D-A (120 prompts, six losing R1 tasks, 27648 cells, stats f16):
- Δ vs dense: mv −15.75 [−22.04, −9.63]; quest −3.38 [−9.08, 2.62]; mv:d2 −18.54 [−25.17, −12.38]; quest:d2 −3.33 [−8.71, 2.08].
- S = +12.38 [6.00, 18.83] (counts); E = −2.79 [−6.04, 0.00] (does not count).
- S counts, so by the PORT-D rule Phase 3 uses the best of the four by Δ: quest:d2. It beats quest by 0.05 points, which is a tie in practice (quest:d2 − quest = +0.04 [−3.12, 3.00]); the rule has no tie-break and is applied as written.
- Checks: D-A dense matches R1 dense token for token on 120/120; D-A mv matches R1 mv token for token on 91/120 (different box and build), with the same score (48.37 vs 48.58).
- Prediction scorecard so far: S counts ✓; E positive but < 3 ✗ (negative); quest:d2 best ✓ (by 0.05); P pending D-B.

**Phase 3 changes.**
- L2 and L3 run with `quest:49152:d2` and `quest:98304:d2` (build 5242a9188), in place of mv.
- Added before L3, so the speed claim matches the configuration whose quality is measured: llama-bench with the same build at 131072 and 262144, 5 reps each: dense, mv:49152 (repeat of a Phase 2 point, as a build check), quest:49152:d2, and at 262144 also quest:98304:d2. Speedup for quest:d2 is reported against the dense run of the same build. No new speed prediction is registered for quest:d2; the L1 predictions (mv) stand as scored.
