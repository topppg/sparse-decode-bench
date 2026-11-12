<!-- first committed in the project repository as ea0a25e at 2026-11-11T21:40:54+01:00 -->

## PORT-MQ4 (2026-11-11): Qwen3-30B-A3B at Q4_K_M on an RTX 5090, speed and RULER at 128k (pre-registered before any PORT-MQ4 data; PORT-MQ results seen)

**Question.** PORT-MQ passed RULER (Δ −0.6) at Q8_0 with 1.41x on an RTX 4090 48 GB. Most users run 4-bit weights on 24–32 GB cards. Does the result carry over to Q4_K_M on an RTX 5090 (32 GB, Blackwell), in speed and quality?

**Engine.** As PORT-MQ (release build e7059e52a; `CMAKE_CUDA_ARCHITECTURES` from the GPU). The fork has never been built or run on Blackwell (sm_120): the same smoke step (quest:12288 vs dense at 65536, ≥ 1.02x) gates the rest.

**Model.** `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`, `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`, SHA-256 6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0 (checked; a mismatch stops the run). KV q8_0.

**Speed and quality.** Exactly as PORT-MQ: same depths, budgets, repetitions, RULER data (the token file must equal PORT-MQ's, SHA-256 adcc34b0…), configs dense and quest:27648, scorer and gate (Δ ≥ −3.0 and 95% lower bound ≥ −6.0).

**Predictions.**
- Smoke: passes.
- S: quest at 131072 is 1.3–1.9x dense (point 1.55). Two effects pull in opposite directions: 4-bit weights make the cache a larger share of each token's reads (higher speedup), while the 5090's higher bandwidth shrinks the dense attention cost relative to the per-layer sparse overhead, which PORT-MQ found to be about half the sparse step (lower speedup).
- Q: PASS, Δ in [−3, +1], about 70% confident.

**Consequence (fixed now).**
- If in before the Reddit post: the post adds the Q4_K_M / RTX 5090 row with its speed and RULER verdict, pass or fail; otherwise it follows as an update comment.
- The bench README gets the result either way, as a row next to PORT-MQ.
- If the smoke step fails, the post and README say the fork does not yet run on Blackwell.
