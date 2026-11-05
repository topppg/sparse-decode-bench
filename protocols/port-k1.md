<!-- first committed in the project repository as ff33afb at 2026-11-03T20:18:56+01:00 -->

## PORT-K1 (2026-11-03): page-level sparse decode on Metal (pre-registered before any K1 timing data; K0 results seen)
**Motivation.** R-PORT-K0 failed on two costs:
- the token-level scorer plus top-k, which cap S at 1.84;
- the scattered gather, at 4× dense cost per key.

Page-level selection (MOIRA 2610.04313) reads contiguous 64-token pages and scores per-page statistics instead of keys. This test asks whether that reaches 2× on the M4 Pro. **Speed only;** quality is MOIRA's published claim and is not tested here.

**Setup.**
- llama.cpp branch `sparse-gqa`, with the sparse n_kv_max cap raised to 32768. Correctness was checked against CPU at 24576 before timing.
- Shape and KV type as in K0: Llama-8B, q8_0, one decode token, GQA bundling on.
- `test-backend-ops perf`, env `SPARSE_K1=1 SPARSE_BLOCK=64`, 3 repeats, median, under `guarded_cmd.sh`. `SPARSE_BLOCK=64` makes the test mask select B/64 random whole 64-cell pages.
- **Grid:** T ∈ {32768, 65536, 131072}; B ∈ {4096, 8192, 12288, 16384, 20480, 24576, 32768}, with B < T.

**Components per layer.**
- `a_dense(T)`: FA, GQA = 1.
- `a_sparse(T, B)`: FA over B cells in whole pages, GQA = 1.
- `s(T)`: page scorer stand-in. `mul_mat` f16 of T/64 page rows × 256 (per-channel mean and variance) per KV head, against G = 4 query columns (q and q², concatenated).
- `sel(T, B)`: `top_k` f32 over T/64 pages per KV head, k = B/64.
- `m(T, B)`: `top_k_mask` with 64-cell blocks, real cache layout.
- Page-statistics upkeep (once per 64 tokens) is ignored.

**Projection:** as in K0. W = 27.8 ms, L = 32, `S = (W + L·a_dense) / (W + L·(a_sparse + s + sel + m))`.

**Budgets, from MOIRA's reported kept fraction.** MOIRA reports 42% at 8k and 15% at 130k; we interpolate log-linearly in T.
- Primary: (131072, 20480), 15.6%.
- Secondary: (32768, 8192), 25%; (65536, 12288), 19%.

**Verdict.**
- **PASS:** S(128k, 20480) ≥ 2.0 and S(32k, 8192) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0.
- **KILL:** S(128k) < 1.5.

**Also reported:**
- the break-even B per T (S = 1);
- the largest B with S ≥ 2 at 128k;
- the per-key cost of the page gather against the token gather (K0) and dense streaming.

**If PASS, next:** an offline quality check of the page mean/variance selector at matched kept fraction, on our PG19 harness (KL at 32k–128k, retrieval), before any integration.
