<!-- first committed in the project repository as bc1cf74 at 2026-11-03T20:45:18+01:00 -->

## PORT-K2 (2026-11-03): a fast sparse FA kernel (pre-registered before any K2 timing data; K0 and K1 results seen)
**Motivation.** In K1 the sparse FA cost 65–85 µs per 1k keys against about 20 for dense, independent of access pattern.
- Code inspection: the sparse path uses `fa_vec_baseline_cfg` (NE = 1 for dk = 128: one KV row per simdgroup step).
- Dense decode on Apple9 (M4 Pro) uses the tuned config (Q = 1, NE = 4: four rows in flight per simdgroup).
- The inner loops are otherwise the same except for the row index.

**Variants.**
- **V1 (primary):** the sparse path uses the tuned config with Q forced to 1. It is env-gated, `GGML_METAL_FA_SPARSE_TUNED=1`, and the bucket is n_kv_max.
- **V2 (built and measured only if V1 is not PASS):** a page-aware sparse kernel that reads one page start per 32-cell chunk and uses dense pointer arithmetic inside the page.

**Setup.**
- As in K1: page masks (`SPARSE_BLOCK=64`), the same grid, components and projection, `port_k1_analyze.py` logic, and results in `results/port_k2/`.
- Correctness against CPU is checked before timing: the q8_0 cases at 4176 / 8272 / 16384 / 24576, with token and page masks.
- Secondary: the K0 token-level grid rerun with V1, labelled [secondary].

**Verdict (V1, then V2 if run):**
- **PASS:** S(128k, 20480) ≥ 2.0 and S(32k, 8192) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0.
- **KILL:** S(128k) < 1.5.

**Also reported:** µs per 1k keys of sparse vs dense.
