<!-- first committed in the project repository as 4ee60c0 at 2026-11-05T16:57:06+01:00 -->

## PORT-C3 (2026-11-05): last CUDA engineering round, two selection kernels fixed (pre-registered before any timing of the fixed build; R-PORT-C, R-PORT-C2 and the exploratory stage timings X-PORT-C-DRY seen)

Seen. Stage costs on `vastai8` (48 GB RTX 4090), 3B, KV f16, in ms per token over dense at 8k / 32k / 128k:
- stats update: 0.86 / 0.96 / 1.03
- scoring: 0.10 / 0.59 / 1.01
- log-softmax and group max: 0.72 / 0.97 / 2.78
- top-k: 0.12 / 0.39 / 1.35
- repeat and mask: 1.02 / 1.42 / 2.26

Full sparse: 7.85 / 9.36 / 17.29 ms, against dense 5.75 / 8.61 / 20.72.

Change (llama.cpp `sparse-cuda` d96e8140f), in CUDA kernels only; the graph and the selection rule are unchanged:
- `pool_1d` with one window per row (OW = 1) uses a block reduction per row. Before, one thread scanned the whole row: 2048 pages at 128k.
- `top_k_mask` runs as a fill kernel with many blocks per row, then a scatter kernel in which 64 threads write each picked page's cells. Before, there was one block per row: 24 blocks at decode.

Gates, all on `vastai8`:
- **K-gate (correctness):** test-backend-ops `-o POOL_1D` and `-o TOP_K_MASK` on CUDA0 against CPU all pass. PORT-C C1 (sparse FA eval, 4 runs) still passes.
- **F-gate (fidelity unchanged):** `llama-sparse-tail`, 3B, T = 8192, 4 books, mv:2048 and quest:2048, f32 page cache. Mean KL within ±1% relative of the PORT-C C4a values (0.017462, 0.017085).
- **S-gate (speed, the decision):** `llama-bench` as in PORT-C C3, 3B, KV f16, depths 8k / 32k / 128k, budgets 2048 / 4096 / 28672, dense and sparse in the same run.
  - Prediction: S(128k) = 1.6, band [1.45, 1.75]; S(32k) in [0.95, 1.2]; S(8k) in [0.75, 0.95].
  - **Engineering success if S(3B, 128k) ≥ 1.5.** Otherwise the engineering phase ends with the current result.

There are no further rounds after this one, whatever the outcome. The next steps are the write-up and, if wanted, the Metal test.
