<!-- first committed in the project repository as 6daa19f at 2026-11-05T11:49:58+01:00 -->

## PORT-C (2026-11-05): page-sparse decode on CUDA (pre-registered before any CUDA data; all Metal PORT-K/E results, the 1.5B llama-bench smoke run and a Metal run of the new test cases seen)

Question: does the page-sparse decode built for Metal give the same selection, the same KL and a tokens/s gain on a consumer NVIDIA GPU, and does a cost model calibrated at short context predict that gain at long context?

Engine: llama.cpp branch `sparse-cuda` (4b281fd64, private backup). The in-graph selection (PORT-E1/E2) is unchanged. CUDA changes:
- The vector FA kernel accepts one mask per q head (`mask->ne[2] > 1`), decode only. Upstream CUDA rejects such masks.
- `GGML_CUDA_FA_SPARSE_PAGED=1`: a prepass builds, per q head, the list of visible 32-cell chunks (a chunk is visible iff its first cell is); the vector kernel then gives each warp one chunk from the list. This is the CUDA analogue of Metal K3.

Hardware: one RTX 4090 24 GB (Ada, sm_89), CUDA 12.x, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`. If no 4090 is available, another Ada card, stated in the results; the numeric bands below are for the 4090. One GPU job at a time.

Dense baseline everywhere: the same binary without the sparse env, so the stock kernel choice (the vector kernel for these one-token decode shapes on Ada).

**C1 correctness (gate before C2–C4).** `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0` with `SPARSE_CUDA_EVAL=1 SPARSE_MASK_HEADS=1`: 16 cases (gqa 3 and 4, kv 8192 and 32768, n_kv_max 1056 and 4096, f16 and q8_0 KV), each under `SPARSE_BLOCK` ∈ {32, 64} × `GGML_CUDA_FA_SPARSE_PAGED` ∈ {0, 1}. PASS if all 64 pass the harness tolerance against CPU. Negative control: PAGED=1 with token-level masks (`SPARSE_BLOCK` unset, not chunk-aligned) must fail at least 8 of 16, showing the paged path is active.

**C2 kernel speed.** `test-backend-ops perf` with `SPARSE_CUDA_PERF=1`: dense runs plain; sparse runs with `SPARSE_MASK_HEADS=1 SPARSE_BLOCK=64 GGML_CUDA_FA_SPARSE_PAGED=1`. kv ∈ {8k, 16k, 32k, 64k, 128k}, B ∈ {2048, 3072, 4096, 6144, 8192, 12288, 14336, 16384, 24576, 27648, 28672, 32768} (B < kv), gqa 3 and 4, f16 and q8_0. Median of 3 invocations.
- H-C2a: the sparse per-key cost (slope of time over B at kv = 128k) is within 1.5× of the dense per-key cost (slope over kv), in each of the 4 (gqa × KV type) cells.
- H-C2b: S_attn(128k, 16384) = t_dense / t_sparse ∈ [4, 8] in each cell.
- PASS both hypotheses in all 4 cells; PARTIAL in 2–3 cells; KILL in fewer than 2.

**C3 end-to-end tokens/s.** `llama-bench -fa 1 -p 0 -n 64 -r 5 -d T`, decode tokens/s, dense vs sparse (`LLAMA_SPARSE_PAGES=B LLAMA_SPARSE_SCORE=mv LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`).
- Models: Llama-3.2-3B q8_0, KV f16 (the PORT-E model); Llama-3.1-8B q8_0 (`unsloth/Meta-Llama-3.1-8B`, the PAGE-Q checkpoint, converted the same way), KV q8_0 (f16 KV does not fit 24 GB at 128k).
- B per T: 2048 at 8k (the E1 budget); otherwise the PAGE-Q mv64 r*(T) at KL 0.02 rounded up to a multiple of 1024. 3B: 4096 / 14336 / 28672 at 32k / 64k / 128k. 8B: 6144 / 12288 / 27648.
- Calibration depths: 8k and 16k (3B B = 2048 and 3072; 8B B = 2048 and 3072). Test depths: 32k, 64k, 128k.
- Cost model (H-C3a, the main prediction): t_sparse(T, B) = t_dense(T) − L·[a_dense(T) − a_sparse(T, B)] + o0 + o1·T. Here a_dense, a_sparse are the C2 per-call FA times at exactly these (T, B) points (same box; gqa 3 and f16 KV for the 3B, gqa 4 and q8_0 KV for the 8B), L the layer count (28 and 32), and o0, o1 the selection overhead per token fitted exactly from the two calibration depths. PASS if the predicted sparse tokens/s is within ±10% of the measured value at every test depth, for both models. Fitted o0, o1 are reported.
- H-C3b (bands, from bandwidth arithmetic before any CUDA data): S = sparse / dense tokens/s.
  - 3B: 8k [0.8, 1.1]; 32k [1.1, 1.6]; 64k [1.3, 2.0]; 128k [1.7, 2.5].
  - 8B: 32k [0.9, 1.2]; 64k [1.0, 1.4]; 128k [1.2, 1.7].
- Usefulness verdict: PASS if S(3B, 128k) ≥ 1.5 and S(8B, 128k) ≥ 1.2; PARTIAL if only one holds; KILL if neither.

**C4 fidelity on CUDA** (`llama-sparse-tail`, as PORT-E1; f32 page cache, PAGED=1).
- C4a backend equivalence: 3B, T = 8192, 4 books, mv:2048, quest:2048, mv:4096, quest:4096. PASS if each config's mean KL (over 4 books) is within ±5% relative of the Metal value in `results/port_e1/e2_full_*.jsonl` (mv:2048 0.0168, quest:2048 0.0165, mv:4096 0.0061, quest:4096 0.0041).
- C4b = PORT-E1 gate E1c run on CUDA instead of Metal (backend change only; gates, model, books, budgets, f32 cache unchanged).
- C4c 128k fidelity, 3B, KV f16, 4 books, mv64 B ∈ {8192, 16384, 32768}: per book and budget the ratio of engine KL to PAGE-Q KL at 131072. Same rule as E1c: median ratio in [0.8, 1.25] and every ratio in [0.5, 2.0]. Reported, no gate: quest64 at the same budgets and engine r* by the PAGE-Q crossing rule.
- Not in this protocol: 8B fidelity at 128k with KV f16 (needs ≥ 32 GB of VRAM).

Order: C1, then C4a, C2, C3, C4b, C4c. A failed C1 stops everything until it is fixed; the fix and the failure are reported.
