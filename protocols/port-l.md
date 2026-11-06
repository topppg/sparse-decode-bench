<!-- first committed in the project repository as 43cc46a at 2026-11-06T09:59:11+01:00 -->

## PORT-L (2026-11-06): decode at 256k context, and whether the quality-safe budget is a fixed number of cells (pre-registered before any PORT-L measurement; setup running; PORT-R R1–R3, PORT-R2 U and R5 seen; R4 running, not read)

**Question.** At 128k the quality-safe budget (R4 will tell which) gives a modest speedup. If what the model needs is a roughly fixed number of cells (the relevant pages plus their neighbourhood) rather than a fixed fraction of the context, the same budget at 256k keeps quality and reads half the fraction, so the speedup grows with context. PORT-L tests this, and tests the cost model 8× beyond its calibration range.

**Box and engine.**
- `vastai14`: RTX 4090 48 GB, CUDA 13.2.
- llama.cpp `sparse-cuda` 2dddd1fa3 = 8213b023d plus microbenchmark cases at 262144 cells and budgets up to 131072 (test-only change).
- Env as in all speed runs: `LLAMA_SPARSE_SCORE=mv LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`, budget via `LLAMA_SPARSE_PAGES`.
- Model: `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct` (same architecture as Llama-3.1-8B), converted f16 then q8_0 by `setup_port_l.sh`, SHA-256 recorded. KV q8_0.

**B\*.** The smallest R4 budget that passes its gate; 65536 if neither passes. Fixed by R4 before any L2/L3 data.

**L1, speed prediction at 128k and 256k (gate).**
- Phase 1: attention microbenchmarks (`SPARSE_CUDA_PERF`, dense and group-shared g = 4, 3 runs, medians); `llama-bench -n 64` dense at T ∈ {8192, 16384, 32768, 131072, 262144} (10 reps ≤ 32k, 5 above); sparse at the calibration points 8192:2048, 16384:3072, 32768:6144 (10 reps).
- Predictions by `port_g_predict.py` (calibrated at 8k–32k, as PORT-G), for T = 131072 with B ∈ {27648, 49152, 65536} and T = 262144 with B ∈ {27648, 49152, 65536, 98304, 131072}. Committed before Phase 2.
- Phase 2: sparse `llama-bench` at those 8 (T, B) points, 5 reps.
- **Gate:** PASS if all 8 predictions are within ±10% of the measured S; PARTIAL if at least 6.
- **Prediction (stated now, before Phase 1):** S(262144, 65536) ∈ [1.4, 1.9]; S(262144, B) > S(131072, B) for every B in both lists.

**L2, RULER subset at 128k and 256k (the fixed-budget hypothesis).**
- Tasks: the six that lost points in R1: niah_multikey_1–3, niah_multivalue, niah_multiquery, vt. 20 samples per task, seed 42, prepared with RULER db43f92 for each `max_seq_length` (131072 and 262144), the UltraLong tokenizer, and template `meta-llama3` if RULER's template.py has it at that commit, otherwise `base` (recorded). The other R1 tasks are excluded: single-needle tasks were at ceiling, and cwe at the floor.
- Prompts, `max_new` and scoring as R1 (`port_r_tokens.py --tasks`, `port_r_score.py` with the UltraLong tokenizer).
- Configs: 128k: dense, mv:B\*. 256k: dense, mv:B\*, mv:2B\*.
- Δ is the six-task mean, so it is larger in magnitude than a 13-task Δ for the same loss and is not compared to R1/R4 numbers directly.
- **H (fixed absolute budget) is supported** if Δ(256k, B\*) − Δ(128k, B\*) ≥ −3 points (difference of the two paired Δs; 95% CI by independent bootstrap, reported). **H is refuted** if that difference is < −3 and its 95% upper bound is < 0. Otherwise inconclusive.
- Informative only if the dense six-task mean at 256k is ≥ 30; otherwise L2 is reported as uninformative.
- mv:2B\* at 256k (the fixed-fraction alternative) is reported alongside.
- **Prediction:** H supported.

**L3, KL at 128k and 256k (reported, no gate).** `llama-sparse-tail --tokens`, 4 PG19 books per length if the validation and test splits have them (else fewer, reported), 256 teacher-forced tokens, `LLAMA_SPARSE_PAGE_STATS=f32`: dense, mv:B\*, mv:2B\* at each length.

Order: setup, L1 Phase 1, commit predictions, L1 Phase 2, L3, L2 (128k, then 256k).
