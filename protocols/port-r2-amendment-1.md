<!-- first committed in the project repository as f9b9658 at 2026-11-06T09:45:55+01:00 -->

## PORT-R2 amendment 1 (2026-11-06, after U and R5 results, before any R4 data was read)

**Results so far.**
- U1 PASS (224/224). U2 PASS (38/38). U4 PASS (no POOL_1D case slower; 4.3–65× faster). Feature branch FLASH_ATTN_EXT 4148/4148, POOL_1D 216/216, TOP_K_MASK 38/38.
- **U3 FAIL:** decode cases (nb = 1) are 1.2–2.0× faster, but 4 of 8 nb = 2048 cases are 1.06–2.2× slower than master. The `top_k_mask` PR is not opened at be1b3f418.
- R5 (4090, 8B, 128k): 36.9 tok/s dense, 45.5 at 49152 (1.23×), 40.8 at 65536 (1.11×).

**Fix.** `b16c04c05` (PR branch) and `55dee4e39` (feature branch): when `ggml_nrows(dst) >= nsm`, the op runs master's fused one-block-per-row kernel unchanged; only fewer rows take the split path. An eval case with 512 rows is added, so both paths are tested on any GPU.

**U3b (gate for the PR, replaces U3):** on the same box after R4, TOP_K_MASK eval 0 FAIL, and `perf -o TOP_K_MASK` median of 3: no case more than 5% slower than master, and the nb = 1 cases at n_kv 131072 at least 1.3× faster. The feature branch's TOP_K_MASK eval is rerun as well (0 FAIL, reported).
