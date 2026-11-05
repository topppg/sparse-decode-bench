<!-- first committed in the project repository as a9ec3c7 at 2026-11-04T11:18:29+01:00 -->

## PORT-E2 (2026-11-04): cached page statistics are exact (pre-registered before any E2 data on the protocol model; development runs on Qwen2.5-1.5B seen)

Engine: `LLAMA_SPARSE_PAGE_STATS=f32|f16` keeps [mean, var, max, min] of K per 64-cell page in the KV cache. A page is refreshed when a ubatch completes it and at every decode step in which it is the newest eligible page. `LLAMA_SPARSE_STATS=full` is the stage-A reference (recompute from the whole cache).

Seen on Qwen2.5-1.5B q8_0, T = 8192, 1 book: f32 cache bit-identical logits to full for mv and quest; f16 cache bit-identical for quest, mv mean KL 2.3e-5 (max 1.4e-3).

Gates on Llama-3.2-3B (q8_0 weights, KV f16), T = 8192, 4 books, configs mv:2048, quest:2048, mv:4096, quest:4096:
- E2a f32 cache: logits bit-identical to full, every config and book.
- E2b f16 cache: quest bit-identical; mv mean KL vs full ≤ 1e-4 per config (mean over 4 books).
