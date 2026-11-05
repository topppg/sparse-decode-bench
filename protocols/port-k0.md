<!-- first committed in the project repository as cf8e553 at 2026-11-03T19:26:40+01:00 -->

## PORT-K0 (2026-11-03): can training-free sparse decode reach 2× on Metal? (pre-registered before any timing data)

**Question.** On the M4 Pro, how fast is the whole sparse attention path (score, select, build the mask, sparse FA with GQA bundling) compared with dense FA with GQA bundling? This is the fair baseline: upstream draft PR #30728.

**Setup.**
- llama.cpp branch `sparse-gqa`: `sparse-decode-design`, plus the 3 commits of PR #30728, plus one host change that allows GQA bundling on the sparse path for single-token decode (`ne01 == 1`).
- Correctness was checked against CPU before any timing: 6/6 OK at kv = 32k, n_kv_max 4176 / 8272 / 16384, f16 and q8_0.
- **Shape:** Llama-3.1-8B / Qwen3-8B attention (dk = dv = 128, 8 KV heads, G = 4), q8_0 K/V, one decode token. The sparse mask has randomly scattered indices.
- **Grid:** T ∈ {32768, 65536, 131072}; B = n_kv_max ∈ {2128, 4176, 8272, 16384}.
- `test-backend-ops perf`, env `SPARSE_K0=1`, 3 repeats, median µs per op. Runs under `run_scripts/guarded_cmd.sh` (kill if free memory < 20% or swap grows by > 1.5 GB).

**Components, per layer, all measured with existing ops:**
- `a_dense(T)`: FA with n_kv_max = 0, with `GGML_METAL_FA_VEC_GQA=1`. Stock (GQA = 0) is also reported.
- `a_sparse(T, B)`: FA with n_kv_max = B and GQA = 1.
- `s(T)`: the scorer, a 4-bit stand-in: `mul_mat` q4_0, T × 128 K per KV head against the G = 4 query columns. It is an upper bound for the planned 2-bit scorer.
- `sel(T, B)`: `top_k` f32 over T per KV head (8 rows), k = B − 80, rounded down to the grid {2048, 4096, 8192, 16384}.
- `m(T, B)`: `top_k_mask` building per-KV-head masks from B cell indices.

**Projection (end-to-end decode per token, Llama-3.1-8B Q4_K_M, L = 32).**
- `t_dense = W + L·a_dense`.
- `t_sparse = W + L·(a_sparse + s + sel + m)`.
- `W = 4.92 GB / 176 GB/s = 27.8 ms`, using the effective weight bandwidth from R-HEADROOM.
- Other per-token costs are ignored on both sides.
- `S(T, B) = t_dense / t_sparse`.

**Primary cell:** (T = 131072, B = 8272). The measured 8B r_KV* at 128k is about 7.7k plus sink and window. Secondary cells: (32768, 2128) and (65536, 4176).

**Verdict.**
- **PASS:** S(128k) ≥ 2.0 and S(32k) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0. Proceed only if a bandwidth bound for a 2-bit scorer plus threshold selection closes the gap; that bound is reported as [expl].
- **KILL:** S(128k) < 1.5.

**Also reported:**
- `a_dense / a_sparse` (kernel only);
- the sparse speedup from GQA bundling (`a_sparse` at GQA 0 vs 1);
- the component breakdown.

**Caveats:**
- The test mask has one row broadcast over heads, so the index-gather pass is 1/8 of a per-KV-head setup. It is small next to KV reads and is reported as is.
- The random index pattern is a worst case for locality.
- No quality is measured here; quality comes from R-PREDICT-128K (kq2 within 10% of the oracle r_KV*).
