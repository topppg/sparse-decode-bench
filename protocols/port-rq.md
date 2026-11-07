<!-- first committed in the project repository as cd1e725 at 2026-11-07T18:03:41+01:00 -->

## PORT-RQ (2026-11-07): RULER with the quest score at the speed-table budgets on Qwen3-8B and Llama-3.2-3B (pre-registered before any PORT-RQ data; PORT-R, R3, PORT-L, V-FEAT-QUEST and V-TRY-IT seen)

**Question.** The release speed table gives Qwen3-8B 1.81x at 27648 cells and Llama-3.2-3B 1.35-1.55x at 28672 cells (128k). RULER with quest, the released default, was measured only on Llama-3.1-8B (R3). Do these two operating points pass the same gate?

**Engine.** The public release build: `topppg/llama.cpp` `sparse-decode` e7059e52a, `llama-sparse-tail --gen`. For each prompt it prefills densely once, then decodes greedily per config from the same cache. The run stops if the cloned commit is not e7059e52a. One CUDA GPU with at least 24 GB; quality only, no speed measured.

**Q, Qwen3-8B.**
- Model: Qwen3-8B-Base with Qwen's YaRN setting (factor 4, original 32768, `max_position_embeddings` 131072), q8_0, as in `setup_port_g.sh`. SHA-256 recorded.
- KV q8_0. Configs: dense, quest:27648.
- Data: RULER, all 13 tasks, 20 samples per task, seed 42, `max_seq_length` 131072, `base` template, RULER commit db43f92 (as R1), tokenizer `Qwen/Qwen3-8B-Base`. Tokens from `port_r_tokens.py --model Qwen/Qwen3-8B-Base` (the tokenizer's default special tokens).

**S, Llama-3.2-3B.**
- Model: Llama-3.2-3B q8_0, SHA-256 068ac283... (as PORT-R).
- KV f16, as in the speed table. Configs: dense, quest:28672.
- Data: the 260 R1 prompts (`ruler8b.i32`, SHA-256 e2fb9ecd..., checked on the box). Llama 3.2 uses the Llama 3.1 tokenizer.

**Scoring.** `port_r_score.py`, as R1 and R3. Mean over the 13 tasks; paired bootstrap with 10000 resamples. Outputs are decoded with each model's tokenizer.

**Gate (per model, as R1 and R3).** Δ = quest - dense ≥ -3.0 points, and the 95% lower bound ≥ -6.0.

**Check (no gate).** If a model's dense mean is below 30, its Δ is reported as weak evidence, because there is little left to lose.

**Predictions.**
- S: PASS, with Δ in [-3, 0]. The budget is about the same fraction of the cache as the R3 budget.
- Q: PASS, with Δ in [-4, 0]; less confident (about 60%). Under mv, Qwen narrowly failed its KL and single-key gates at 27648 (PORT-G), but quest removed most of the multi-needle loss on Llama.

**Consequence (fixed now).**
- PASS: the bench README gives the model's RULER result next to its speed row, and the headline quality sentence covers it.
- FAIL: the README marks that speed row as not quality-checked at that budget and gives the RULER numbers. The headline quality sentence names only passing models. No budget search before the release.
- The result is published either way.

**Order.** Q, then S, on one box. If two boxes are available, Q and S run in parallel, one model each.
