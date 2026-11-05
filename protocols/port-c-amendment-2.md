<!-- first committed in the project repository as d9fb4db at 2026-11-05T12:23:54+01:00 -->

## PORT-C amendment 2 (2026-11-05, after C1 and before any C2–C4 data)

The Vast box (`vastai7`) has an RTX 4090 with 48 GB, a memory-modified AD102 board (sm_89; driver 595.104, CUDA 13.2), not the 24 GB card. Chip and nominal bandwidth are those of the 4090, so the C2 and C3 bands still apply as written. Models, KV types and budgets are unchanged (the 8B keeps KV q8_0), so nothing depends on the extra memory. The 3B q8_0 file rebuilt on the box has the same SHA-256 as the local PORT-E file.

C1 result (seen before this amendment): PASS. 17/17 cases in each of the 4 runs, and the negative control failed 17/17.
