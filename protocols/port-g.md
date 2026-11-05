<!-- first committed in the project repository as a7c7927 at 2026-11-05T17:36:14+01:00 -->

## PORT-G (2026-11-05): a second GPU (RTX 3090) and a second model family (Qwen3-8B), with speedups predicted from short-context calibration (pre-registered before any 3090 or Qwen3 data; all PORT-C, C2 and C3 results and a retrospective check of the procedure on the 4090 seen)

Question: does the PORT-C3 result carry over to another consumer GPU and another model family? Is the long-context speedup predictable from measurements at ≤ 32k on the target GPU?

**Engine and hardware.**
- llama.cpp `sparse-cuda` 86c006aca. That is d96e8140f plus a `--kv-type` option in `llama-sparse-tail`; no kernel change.
- One RTX 3090 24 GB (Ampere, sm_86), on Vast. The exact board is stated in the results.

**Models (`setup_port_g.sh`).**
- Llama-3.2-3B and Llama-3.1-8B q8_0. Their SHA-256 must equal the PORT-C files.
- Qwen3-8B-Base, converted with Qwen's YaRN setting (`rope_scaling` yarn, factor 4, original 32768; `max_position_embeddings` 131072), then q8_0. Its SHA-256 is recorded.

**Retrospective check, seen (4090, exploratory).** Calibrating the cost model on 8k and 32k predicts S(128k) = 1.318 for the 8B (measured 1.313) but 1.25 for the 3B (measured 1.55). The 3B's 32k sparse point was noisy (±6.5%). Hence three calibration depths, 10 repetitions, and a least-squares fit.

**G1, predicted speedups (`run_port_g.sh`, `src/port_g_predict.py`).**
- Phase 1 on the 3090:
  - FA microbenchmarks: dense, plus group-shared page masks (`SPARSE_MASK_GROUP`, gqa 3 and 4, f16 and q8_0);
  - `llama-bench` dense decode tokens/s at 8k / 16k / 32k / 64k / 128k (r = 10 up to 32k, r = 5 above);
  - sparse at 8k / 16k / 32k only (r = 10).
- Models and KV types: 3B with KV f16; 8B and Qwen3 with KV q8_0.
- Budgets: 3B 2048 / 3072 / 4096 / 14336 / 28672; 8B and Qwen3 2048 / 3072 / 6144 / 12288 / 27648.
- Layers: 3B 28, 8B 32, Qwen3 36. gqa: 3, 4 and 4.
- `port_g_predict.py` gives the predicted sparse tokens/s at 64k and 128k:
  - t_sparse(T) = t_dense(T) − L·[a_dense(T) − a_sparse(T, B)] + o0 + o1·T;
  - o0 and o1 are fitted by least squares on the three calibration depths;
  - a are the microbenchmark times.
- The predictions are committed before phase 2 runs.
- Phase 2: sparse at 64k and 128k (r = 5).
- **H-G1: |pred_S / meas_S − 1| ≤ 10%** at 64k and 128k for each of the 3 models (6 points). PASS if all 6 hold; PARTIAL if 4–5; FAIL if fewer than 4.
- Reported against the PORT-C bars, not gated: S(128k) for the 3B (≥ 1.5), the 8B (≥ 1.2) and Qwen3 (≥ 1.2).

**Q1, Qwen3 fidelity at 128k (in-engine; there is no Python reference for Qwen3).**
- `llama-sparse-tail`, T = 131072, KV q8_0, f32 page cache, the same 4-book `book_segments` call tokenised with Qwen's tokenizer (book identity may differ from Llama's).
- Configs: dense, mv:{8192, 16384, 27648, 32768}, quest:27648.
- **Gate: mean KL at mv:27648 ≤ 0.03.** That is the Llama budget rule carried over unchanged.
- Reported: the KL curve and r*(0.02) by the PAGE-Q crossing rule.

**Q2, Qwen3 retrieval at 128k.**
- The 40 PAGE-Q trials (single and multikey), rebuilt by `e2_ret_tokens.py` with Qwen's tokenizer.
- Configs: dense, mv and quest at {4096 … 32768}, and mv:27648.
- **Gate: for each task with ≥ 10 dense-correct trials, the mv rate at 27648 is ≥ 0.9 over dense-correct trials.** A task with fewer than 10 is reported as uninformative.

Order: setup, phase 1, predictions committed, phase 2, analysis. RULER is not part of this protocol.
