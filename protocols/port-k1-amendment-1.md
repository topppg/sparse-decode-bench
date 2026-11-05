<!-- first committed in the project repository as a5e4cb1 at 2026-11-03T20:27:05+01:00 -->

## PORT-K1 amendment 1 (2026-11-03, before any K1 timing data; only the correctness gate had run)
- **The f16 sparse FA at n_kv_max = 24576 misses the harness NMSE threshold:** 5.2–5.5e-4 against 5e-4.
  - The miss is the same with GQA bundling off or on, and with token or 64-cell page masks. It is f16 accumulation error over about 24k keys in the existing sparse kernel, not our change.
  - All q8_0 cases pass, including 24576.
- K1 times q8_0 only, so the gate now requires the 4 q8_0 eval cases to pass.
