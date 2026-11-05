<!-- first committed in the project repository as c31a1a6 at 2026-11-05T12:38:55+01:00 -->

## PORT-C2 (2026-11-05): 8B fidelity and retrieval at 128k inside llama.cpp on CUDA (pre-registered before any C2 data; PORT-C C1, C4a and 2 of 3 C2 timing runs seen)

Question: does the in-engine page rule reproduce PAGE-Q's 128k fidelity on the primary model (Llama-3.1-8B), and does in-engine retrieval at 128k need the same budget PAGE-Q found? Both are numbers PAGE-Q's GO decision rests on; so far only the Python reference has produced them.

Engine and box: `sparse-cuda` 310d7c858. It adds a retrieval mode to `llama-sparse-tail` (`--ret`); the CUDA kernels are unchanged from PORT-C. Box `vastai7` (48 GB RTX 4090). Weights q8_0 (the 8B file from PORT-C setup), KV f16, f32 page cache, `GGML_CUDA_FA_SPARSE_PAGED=1`.

**C5a 8B fidelity at 128k.** `llama-sparse-tail`, T = 131072, the 4 PAGE-Q books (`books128k.i32`, same tokens for both models), configs dense, mv:{8192, 16384, 32768}, quest:{8192, 16384, 32768}. Per book and budget, the ratio of engine KL to PAGE-Q KL (`pageq_llama31_8b`, 131072). PASS if the median of the 12 mv ratios is in [0.8, 1.25] and every mv ratio is in [0.5, 2.0] (the E1c rule). Reported, no gate: quest ratios, engine r* by the PAGE-Q crossing rule. Note: C4a found CUDA KL about 4e-4 above Metal at 8k. That is under 3% of the PAGE-Q 8B KL at B = 32768 (0.015), so well inside the gate.

**C5b retrieval at 128k, 8B and 3B.** The 40 PAGE-Q trials (single and multikey; 4 books × 5 depths each), rebuilt by `e2_ret_tokens.py` with PAGE-Q's construction code and RNG seed. Dense prefill of the prompt, then the answer tokens teacher-forced one decode step at a time. A trial is correct if every answer token is the argmax.
- Configs: dense, and mv and quest at {4096, 8192, 12288, 16384, 24576, 32768}, plus mv at the PORT-C C3 128k budget (27648 for the 8B, 28672 for the 3B).
- k_ret as in PAGE-Q: the smallest grid budget from which every larger grid budget has rate ≥ 0.9 over dense-correct trials. Grid: {4096, 8192, 12288, 16384, 24576, 32768}.
- Gates, per model and task:
  - (i) dense-correct trials ≥ PAGE-Q's minus 2. 8B: single ≥ 18, multikey ≥ 17. 3B: single ≥ 18, multikey ≥ 16.
  - (ii) engine mv k_ret within one grid step of PAGE-Q's mv64 k_ret. 8B: 12288 single and multikey. 3B: 12288 single, 8192 multikey.
  - (iii) at the C3 128k budget, mv rate ≥ 0.9 over dense-correct trials.
- PASS if (i)–(iii) hold for both tasks and both models. Each failing cell is reported.

Order on the box: after the PORT-C stages and the C4a diagnostic. C5a first, then C5b for the 8B, then the 3B.
