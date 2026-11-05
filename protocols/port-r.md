<!-- first committed in the project repository as 44a9e78 at 2026-11-05T18:18:05+01:00 -->

## PORT-R (2026-11-05): RULER at the operating point, a speed–budget curve, and wall-clock with prefill, on an RTX 4090 (pre-registered before any PORT-R data; PORT-C, C2, C3 seen, PORT-G running in parallel on a 3090)

**Box and engine.**
- `vastai11`: RTX 4090 24 GB.
- llama.cpp `sparse-cuda` 8213b023d. Kernels as in C3 (d96e8140f); `llama-sparse-tail` adds `--kv-type` and `--gen`, a greedy generation mode.
  - `--gen` prefills the prompt minus its last token densely.
  - Then, for each config, it decodes the last prompt token and up to `max_new` greedy tokens. Generation stops early at EOS.
- Models from `setup_port_g.sh`. The SHA-256 of each file is recorded, and a mismatch with the PORT-C files is reported.

**R1, RULER at 128k, Llama-3.1-8B q8_0, KV q8_0.**
- Data: NVIDIA RULER synthetic, all 13 tasks (niah_single_1–3, niah_multikey_1–3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2).
  - 20 samples per task, seed 42, `max_seq_length` 131072 including generation, `base` template, tokenizer `unsloth/Meta-Llama-3.1-8B`.
  - RULER commit recorded by `ruler_prep.sh`.
  - The prompt is `input + answer_prefix`, tokenised with BOS (`port_r_tokens.py`). `max_new` follows RULER: niah 128, vt 30, cwe 120, fwe 50, qa 32.
- Configs: dense, and sparse mv:27648 (the C3 128k budget) with `LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`, as in the speed runs.
- Scoring (`port_r_score.py`): RULER's own metrics. string_match_all for niah, vt, cwe and fwe; string_match_part for qa. RULER's postprocessing applies. The score is the mean over the 13 tasks.
- **Gate (non-inferiority):** Δ = sparse − dense ≥ −3.0 points, and the 95% paired bootstrap lower bound ≥ −6.0.
- Prediction: Δ ∈ [−3, +1]. The two tasks with the largest drop are among {cwe, fwe, niah_multikey_3, niah_multivalue, niah_multiquery}, the tasks that need many positions.
- Reported: per-task scores for both configs.

**R2, speed–budget curve at 128k (reported, no gate).**
- 3B with KV f16 and 8B with KV q8_0. `llama-bench -d 131072 -n 64 -r 3`: dense, and sparse at B ∈ {8192, 12288, 16384, 27648 (8B) or 28672 (3B), 32768, 49152}.
- To be read together with the engine KL (C4c, C5a) and retrieval rates (C5b) at the same budgets.

**R3, wall-clock with prefill (reported, no gate).**
- `llama-bench -pg 131072,1024 -r 2`: dense vs sparse at the C3 budget, for the 3B (KV f16) and the 8B (KV q8_0).
- Reported as end-to-end tokens/s and seconds for a 128k prompt plus 1024 generated tokens.
- Expectation, stated now: the end-to-end gain is well below the decode-only S, because prefill is dense and dominates.

Order: R1, then R2, then R3.
