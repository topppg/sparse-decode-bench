<!-- first committed in the project repository as 8fbf5f3 at 2026-11-03T21:06:51+01:00 -->

## PORT-K3 (2026-11-03): page-level index gather (pre-registered before any K3 timing data; K2 results seen)
**Motivation.** K2 V2 leaves a fixed cost that grows with T (about 40 µs at 32k, about 115 µs at 128k). One source is the index-gather pass, which scans all T mask cells in one threadgroup per row.

**V3 = V2 + `kernel_flash_attn_ext_vec_idx_pages`.**
- It reads only the first cell of each 32-cell chunk and writes the chunk start at list slot 32·rank.
- Selected with `GGML_METAL_FA_SPARSE_PAGED=1`.
- Correct against CPU for q8_0 at all four budgets; f16 is OK except the known 24576 rounding case.

**Setup, components, projection and verdict rules:** identical to K1 and K2. Page masks, results in `results/port_k3/`.
- **PASS:** S(128k, 20480) ≥ 2.0 and S(32k, 8192) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0.
- **KILL:** S(128k) < 1.5.
