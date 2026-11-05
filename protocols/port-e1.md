<!-- first committed in the project repository as fb55a5d at 2026-11-04T10:51:17+01:00 -->

## PORT-E1 (2026-11-04): in-engine page selection in llama.cpp is correct (pre-registered before any in-engine data; PAGE-Q and K3 results seen)

Question: does page selection inside the llama.cpp graph (Metal, page-gather FA kernel from K3) select the pages PAGE-Q selects and give the KL PAGE-Q measured? This is a correctness gate before any tokens/s claim. Stage A only: page statistics are recomputed from the whole K cache each decode step (no speedup expected); the incremental page cache is PORT-E2.

Engine: branch `sparse-gqa` of the local llama.cpp clone. Env `LLAMA_SPARSE_PAGES=<budget B>`, `LLAMA_SPARSE_SCORE=mv|quest|mean`, decode steps with one token only. Same rule as `t4_select.select_mask` block mode (b = 64): eligible pages 1 ≤ p < win_blk with win_blk = max(0, ⌊(pos − 63)/64⌋); per-query-head page score (mv: s·μ + ½ s²·σ², quest: s⁺·max + s⁻·min, mean: s·μ, with s = q·kq_scale); per-head log-softmax over eligible pages; max over the G heads of the group; top nb = ⌊(B − 80)/64⌋ pages per KV head; keep page 0, the picked pages, and cells win_blk·64 … pos. Metal env `GGML_METAL_FA_SPARSE_TUNED=1 GGML_METAL_FA_VEC_GQA=1 GGML_METAL_FA_SPARSE_PAGED=1`.

Model and data: unsloth/Llama-3.2-3B converted with `convert_hf_to_gguf.py` to f16 GGUF, KV cache f16, flash attention on. The 4 PAGE-Q fidelity books (same `book_segments` call, length 131072, validation split), truncated to T. Dense prefill of T − 256 tokens, then 256 teacher-forced single-token decode steps, as in P-T4. KL(dense ‖ sparse) per row in float64 over the full vocabulary, both runs in-engine.

Gates (all must pass; any failure is reported and fixed before E2):
- E1a exact selection, T = 8192, book 0, layers {0, 9, 18, 27}, decode steps {0, 255}, B ∈ {1024, 4096}, mv and quest. Python recomputes page statistics from the dumped f16 K (cast to f32) and the picks from the dumped q in float64. PASS if statistics max abs error ≤ 1e-3 and the picked page set equals the reference set, except for pages whose reference group score is within 1e-3 of the nb-th score.
- E1b machinery floor, T = 8192, 4 books: with B ≥ T (every eligible page picked) mean KL ≤ 1e-4; page kernel (PAGED=1) vs token kernel (PAGED=0) at B = 2048, mean KL between them ≤ 1e-5.
- E1c fidelity vs PAGE-Q, T = 32768, 4 books, mv64, B ∈ {1024, 2048, 4096, 8192}: per book and budget the ratio of engine KL to PAGE-Q KL (`pageq_llama32_3b` summary). PASS if the median of the 16 ratios is in [0.8, 1.25] and every ratio is in [0.5, 2.0]. Reported, no gate: quest64 at the same cells, and engine r* by the PAGE-Q crossing rule.

Memory: everything under `guarded_cmd.sh`. If T = 32768 with f16 weights is killed, rerun E1c with q8_0 weights (KV still f16) and say so; if that is also killed, E1c waits for a bigger machine.
