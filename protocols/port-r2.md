<!-- first committed in the project repository as 63361a7 at 2026-11-06T07:45:02+01:00 -->

## PORT-R2 (2026-11-06): RULER at larger budgets, and re-validation of the two upstream kernel PRs, on the RTX 4090 (pre-registered before any PORT-R2 data; all PORT-R R1/R2/R3 results, both boxes, and V-UPSTREAM seen)

**Why.** R1 failed at 27648 cells (Δ −7.4, CI [−11.0, −4.1]), with the loss in multi-needle tasks. R2 shows decode is still faster at larger budgets (8B at 128k, 49152 cells: 1.42× on the 3090, 1.24× on the 4090). This test finds whether a budget exists where RULER is non-inferior, which would turn the speed–quality dial into a recommended operating point.

**Box and engine.**
- `vastai11` (RTX 4090 24 GB), the R1 box, restarted with its disk.
- Engine unchanged from R1: llama.cpp `sparse-cuda` 8213b023d, same build, same model file (SHA-256 recorded again), same `ruler8b.i32` token file (SHA-256 recorded) and RULER metadata.
- No code change to the selection path. The proposed page-stats refresh fix is not part of this test.

**R4, RULER at 128k, Llama-3.1-8B q8_0, KV q8_0.**
- Data, prompts, `max_new`, `--gen` mode, env (`LLAMA_SPARSE_SCORE=mv LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`) and scoring (`port_r_score.py`, 10000-sample paired bootstrap): all exactly as R1.
- Configs in one pass (one prefill per sample): dense, mv:49152, mv:65536.
- **Gate per budget (non-inferiority, as R1):** Δ = sparse − dense ≥ −3.0 points, and the 95% paired bootstrap lower bound ≥ −6.0.
- **Recommended operating point:** the smallest tested budget that passes, reported with its measured decode speedup at 128k on both GPUs. If neither passes, there is no recommended budget at 128k for this model; the result is reported as a curve: Δ at 27648, 49152, 65536.
- **Prediction:** Δ(49152) ∈ [−6, −1] (most likely a narrow fail); Δ(65536) ∈ [−4, 0]; Δ is non-decreasing in budget within 1 point. niah_multivalue keeps the largest drop at 49152.
- **Check (no gate):** dense outputs are compared token by token with R1 dense. Any mismatch is reported with its count.
- Reported: per-task scores for all three configs.

**R5, decode speed at 65536 cells (reported, no gate).** `llama-bench -d 131072 -n 64 -r 3`, 8B q8_0 KV, dense and sparse at 49152 and 65536 on the 4090. The 3090 is not re-rented; its 65536 point is reported as predicted by the PORT-G cost model only, labelled as a prediction.

**U, upstream PR re-validation (gates for opening the PRs).**
- Branches as fixed after V-UPSTREAM: `cuda-pool1d-row` 61c176dd8, `cuda-top-k-mask-split` be1b3f418, both on master 6023692b1, built with `CMAKE_CUDA_ARCHITECTURES=89`.
- U1: `test-backend-ops -o POOL_1D -b CUDA0` on `cuda-pool1d-row`: 0 FAIL.
- U2: `test-backend-ops -o TOP_K_MASK -b CUDA0` on `cuda-top-k-mask-split`: 0 FAIL.
- U3: `test-backend-ops perf -o TOP_K_MASK`, median of 3, branch vs master: no case more than 5% slower than master. (V-UPSTREAM found up to 11× slower at nb = 2048 before the fix.)
- U4: `perf -o POOL_1D`, median of 3: no case more than 5% slower than master.
- Also run on `sparse-decode` 394ff1f65: FLASH_ATTN_EXT, POOL_1D and TOP_K_MASK eval, 0 FAIL (reported).
- A PR is opened only if its gates pass.

Order: U (about 30 min), then R5, then R4.
