<!-- first committed in the project repository as 74b0eac at 2026-11-19T10:37:20+01:00 -->

## PF-NT (2026-11-19): RULER at 128k for sparse prefill plus sparse decode with Qwen3.8's no-thinking chat format, BF16 and Q4_K_M weights on the same card (pre-registered before any PF-NT data)

Condensed from the project repository. Code: `scripts/pf_nt/` (decision) and `scripts/pf_e2e/box/in/patches/` (the fork patches, the same seven files).

**Background (seen when written).** An earlier run of the same question with BF16 weights on RULER's raw prompts (PF-BF16) ended NO_DECISION: dense 85.46, combined −7.38 [−10.23, −4.48], dense below its floors on vt (88) and qa_1 (25). Its raw outputs had 18–27 empty outputs per config (19, 18, 24 and 27; 3–5 at Q4_K_M: 4, 5, 3 and 4) and cut-off `<think>` blocks in about 10% of outputs at both precisions. The Q4_K_M run on raw prompts with q8_0 KV (PF-CONF, RTX 5090) had passed: combined −2.38 [−5.00, +0.27]. The likely cause, found after those data, is the raw-prompt format, which also affects the Q4_K_M reference; changing the format and rerunning Q4_K_M is a new protocol. No Qwen3.8 RULER run with a chat template existed. Reviewed before any data (two rounds of an independent model review); the fixes are in this text and the scripts.

**Question.** On Qwen3.8-27B at 128k (`-ub 512`, f16 KV, RTX PRO 6000 Blackwell), with RULER prompts in Qwen3.8's chat format with thinking off:
1. does the combined config `pf:0.1:512:d512+quest:27648:f3` pass the RULER gate at 20 prompts per task with **BF16** weights?
2. does it pass with **UD-Q4_K_M** weights?
Both share the card, the KV type, the prompts and the build; only the weight precision differs.

**Engine.** topppg/llama.cpp at f7d80009d plus patches 0001–0007 by `git am` on a fresh clone, tree **487a7f15479d39eac0b3bf4be0ab46160eca0fb7** and a clean work tree checked before the build. `llama-sparse-tail`, `-ub 512`, KV **f16** for both models (llama.cpp's default, so the weights are the only difference). Every call runs with `LLAMA_GRAPH_REUSE_DISABLE=1` and `LLAMA_SPARSE_PREFILL_GATHER=1` (the sparse gather path), prefill and decode controls cleared first, `GGML_CUDA_PDL` unset.

**Boxes.** Two RTX PRO 6000 Blackwell 96 GB (BF16 Qwen3.8-27B is 54.7 GB of weights); a workstation card, so the numbers are evidence about format and precision, not a consumer-card number. Box A runs pass 1 (prompts 0–9) of both models, box B pass 2 (prompts 10–19) of both, so each pass's BF16 and Q4_K_M rows come from the same card. Gate P: card name contains "RTX PRO 6000", ≥ 90,000 MiB, disk ≥ 140 GB with ≥ 100 GB free (less any model files already present). Power limit, driver and toolkit recorded, not gated. A box that fails gate P is replaced by another RTX PRO 6000 running the same list.

**Models.** `unsloth/Qwen3.8-27B-GGUF` at 4af51b35:
- b38: `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` + `…-00002-of-00002.gguf`, SHA-256 b9966e82…0ba191f, 92e3943c…9edfa02;
- q38: `Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256 1bce504d…dc529.

**Tokenizer.** `Qwen/Qwen3.8-27B` at revision **37561816e66b5b9153b1b64c10968c18c1e6f4a0** for prompts, decoding, scoring and gate F.

**Prompts.** `pf_nt_q38.i32`, SHA-256 **ab921fb5d038ddcfa5076f38f9bfe9249b265e0930d91874396dbec51f265ca7** (meta in `results/cuda_6000_pfnt/inputs/`): the same 260 RULER samples as the earlier Qwen3.8 runs, same ids, generation lengths and references (13 tasks × 20; RULER db43f92, seed 42 is the provenance stated in those earlier protocols, not in this registration).
- Each prompt is RULER's input plus answer prefix, split at the task's last answer-prefix marker ("? The special magic " for niah, " Answer:" otherwise); splitting and re-tokenising gives back the original ids for all 260.
- New prompt = `apply_chat_template([user: input], add_generation_prompt=True, enable_thinking=False)` + the answer prefix verbatim. The template output ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` and has no system turn. Every prompt gains exactly 12 tokens; lengths 111,921–131,049; context at most 131,328.
- Greedy decoding with RULER's generation lengths (niah 128, vt 30, cwe 120, fwe 50, qa 32).

**Configs per prompt** (two shared prefills): `dense`, `quest:27648:f3`, `pf:0.1:512:d512`, `pf:0.1:512:d512+quest:27648:f3`.

**Run.** Cap 39,600 s (11 h) per box from launch, setup included. A unit is (model, pass, task): 10 prompts × 4 configs in one process, validated and committed atomically, one retry on a crash, FAILED on invalid rows or at the deadline. Unit timeout min(5,400 s, time left); a unit ending past the deadline is FAILED and not committed. Order: box A b38 units 1.11, 1.8, 1.0, 1.1–1.7, 1.9, 1.10, 1.12, then q38 in the same order; box B q38 units 2.11, 2.8, 2.0, 2.1–2.7, 2.9, 2.10, 2.12, then b38. qa_1 and vt run first because they failed in PF-BF16, so gate F can be read early.

**Gates.**
- **K** (`test-backend-ops` on the fork): `n_top` exactly 6/6; `n_kv_max` and `n_pages` all pass; the sparse op list all passes.
- **Counts** on every row: 16 layers, 4 KV heads, length ≥ 100,000, sparse rows engaged on all 16 layers with no fallback, dense rows not sparse.
- **U**: an nsys check on both boxes, with b38, before the first unit: the `flash_attn_mask_to_sparse_indices` kernels take > 0 s with the gather and 0 s in the emulation.
- **F** (format screen): per format, the 20 dense outputs of the first two units of the box that runs that format first (b38: box A units 1.11 and 1.8; q38: box B units 2.11 and 2.8). F passes for a format iff no output contains `<think>` and at most 2 are empty or blank; F passes iff it passes for both. It may be read live to stop early (saves box time only); `pfn_decide.py` recomputes it from the rows and the recomputed value decides. If one of these units is missing or invalid, F is unavailable (NO_DECISION, not FORMAT_FAIL).
- **D**: the task-0 dense unit of each model ≥ 80 (b38 1.0, q38 2.0), else stop.
- Empty outputs are valid rows scored 0.

**Decision** (`pfn_decide.py`, per model, first match):

| # | condition | model quality |
|---|---|---|
| 1 | an accepted run dir lacks SETUP DONE, SOURCE OK, a GATE K or GATE U PASS line, the model's UNITS COMPLETE line, or a terminal line | INVALID |
| 2 | merged cohort not exact (13 × 20 × 4, each (id, config) once, counts valid) | INCONCLUSIVE |
| 3 | dense below its floor on any task: 90 on each niah task; vt 80, cwe 60, fwe 70, qa_1 40, qa_2 10 | INCONCLUSIVE |
| 4 | the combined config passes: Δ ≥ −3 and 95% lower bound ≥ −6 (1e-9 tolerance) | PASS |
| 5 | otherwise | FAIL |

The non-niah floors are the best earlier raw-prompt dense score of each task minus 20, rounded down to a multiple of 5; they catch a broken file or format and do not compare the models. A model's rows on a box are accepted when the box's shared gates passed, its UNITS COMPLETE line is present and its cohort is exact; gate D failing makes that model INCONCLUSIVE; a missing (model, pass) is rerun by amendment on a new RTX PRO 6000 box, that list only. `analysis/manifest.json` names one run dir per (model, pass).

Δ = mean over the 13 tasks of each task's mean paired difference (config − dense) × 100; bootstrap 10,000 replicates, seed 0, one shared resample of the 20 prompt indices per task; scoring as `port_r_score.py` (`scripts/pf_e2e/src/port_r_score.py`). The two per-format decisions are primary, each with its own interval, no multiplicity correction; BOTH_PASS needs both. The precision contrast is descriptive.

**Outcome** (first match). A FAIL means non-inferiority was not shown, not that a loss was shown.

| outcome | condition | consequence |
|---|---|---|
| FORMAT_FAIL | gate F failed | Qwen3.8 RULER claims stay as the raw-prompt Q4_K_M run states them; a different scoring setup is discussed before any rerun. |
| NO_DECISION | gate F unavailable, or a model INVALID or INCONCLUSIVE | Fix the cause; one rerun of the affected part, by amendment. |
| BOTH_PASS | b38 PASS and q38 PASS | Public quality statements for Qwen3.8 may name both formats and the chat format; the end-to-end stage proceeds. |
| Q4_ONLY | q38 PASS, b38 FAIL | Public quality statements name Q4_K_M only. Before any full-precision claim, the BF16 loss is examined per task with the project lead. |
| BF16_ONLY | b38 PASS, q38 FAIL | Public Q4 statements carry this result. |
| NEITHER | both FAIL | The sparse-prefill program is reassessed before the end-to-end stage. |

Reported with the outcome, never changing it: the other three configs with their gate, each config per pass, per-task scores, empty outputs and output kinds; b38 Δ − q38 Δ; each model against its raw-prompt run (descriptive only: those differ in prompt format, KV type, card, driver and toolkit, so no difference may be attributed to the format alone); token identity of b38 and q38 dense rows; the gather / dense prefill-time ratio.

**Predictions.**
- P1: gates P, K, U, F and D pass on both boxes; counts hold; both cohorts exact.
- P2: at most 5 empty or blank outputs per config per model; no `<think>` text in any output.
- P3: dense means q38 and b38 each in 86–94; qa_1 dense ≥ 50 for both.
- P4: q38 combined Δ in −4.0 to +0.5 (point −2.0); PASS (p ≈ 0.6).
- P5: b38 combined Δ in −5.0 to +0.5 (point −2.5); PASS (p ≈ 0.5).
- P6: the combined config's largest single-task loss (dense − combined; ties count as held) is cwe in both models (so vt recovers under this format).
- P7: BOTH_PASS (p ≈ 0.35), Q4_ONLY (0.25), NEITHER (0.2), other (0.2).
Each range and its PASS clause are graded separately.

**Not settled by PF-NT:** thinking-on use or other generation lengths; q8_0 KV under this format; other models and weight formats; consumer cards at BF16; a dedicated kernel; conclusions independent of the two boxes' driver, toolkit and power limit.
