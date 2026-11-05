<!-- first committed in the project repository as d6ed78a at 2026-11-05T11:57:00+01:00 -->

## PORT-C amendment 1 (2026-11-05, before any CUDA data; a Codex static review of the CUDA diff seen)

Engine is `sparse-cuda` 66ed860e7. It fixes three review findings outside the paged path:
- the dense mask scan no longer runs on per-head masks (it bounded every head by head 0's last visible key when there are several sequences);
- per-head masks with logit softcap go to the vector kernel only for D = 128 or 256;
- paged mode is compiled for CUDA only, not HIP/MUSA.

C1 gains one case, two sequences with per-head masks (gqa 3, kv 8192, n_kv_max 1056, f16): 17 cases, 68 runs, all must pass. The negative control still needs at least 8 failures. Everything else is unchanged.
