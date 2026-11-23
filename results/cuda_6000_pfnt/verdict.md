# PF-NT verdict (Qwen3.8-27B BF16 and UD-Q4_K_M, KV f16, RULER at 128k in the chat format with thinking off, RTX PRO 6000 Blackwell)

Protocol `protocols/pf-nt.md` and amendment 1 (written after data existed). Decision: `analysis/decision.json` (`pfn_decide.py --late F --late D`), recomputed from the committed rows by `scripts/pf_e2e/decide.sh`: identical except the `vs_raw` fields, which compare with earlier raw-prompt runs not published here. Adapted from the project repository's verdict.

Engine: topppg/llama.cpp f7d80009d plus patches 0001–0007 (`scripts/pf_e2e/box/in/patches/`), tree 487a7f154, checked in both accepted run dirs. `llama-sparse-tail`, `-ub 512`, KV f16 for both models, `LLAMA_SPARSE_PREFILL_GATHER=1`, `LLAMA_GRAPH_REUSE_DISABLE=1`. Prompts `pf_nt_q38.i32` (ab921fb5…, SHA-256 checked on the box): 260 RULER prompts in Qwen3.8's chat template with `enable_thinking=False`. Models b38 = unsloth `Qwen3.8-27B-BF16` (two shards), q38 = unsloth `Qwen3.8-27B-UD-Q4_K_M`, SHA-256 checked (`sha.txt`).

## Runs

| run dir | role | card | power limit | final line | rows |
|---|---|---|---|---|---|
| `vastai38/20261119T103525Z` (A1) | pass 1 (prompts 0–9) of b38, then q38 | RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB, driver 595.99.02, CUDA 13.2 | 450 W | COMPLETE | 1,040 |
| `vastai38/20261119T190800Z` (A2) | pass 2 (prompts 10–19) of q38, then b38: box B's list, rerun on box A under amendment 1 | the same card | 450 W | COMPLETE | 1,040 |

Box B (vastai39) was excluded by amendment 1 after it produced degenerate rows (runs of token id 0) in 13 of its 120 prefills, dense and sparse alike, which did not repeat on identical inputs; its rows never enter the decision and are not published here. All four (model, pass) cohorts therefore come from one card.

## Decision

**Outcome BOTH_PASS** (gate F PASS, b38 PASS, q38 PASS). The combined config `pf:0.1:512:d512+quest:27648:f3` passes the gate (Δ ≥ −3 and 95% lower bound ≥ −6) at both precisions:

| model | dense | combined | Δ | 95% CI |
|---|---|---|---|---|
| b38 (BF16) | 95.77 | 94.17 | **−1.60** | [−3.25, −0.15] |
| q38 (UD-Q4_K_M) | 95.77 | 94.12 | **−1.65** | [−3.25, −0.17] |

Both intervals lie below 0: the combined config costs about 1.6 RULER points at both precisions, a loss inside the tolerance, not "no loss". b38 Δ − q38 Δ = +0.06 (descriptive).

## Gates

- P, K (blocks 6/6, sparse 26/26, pages 128/128, ops 1454/1454) and U (gather 1.45 s, emulation 0 s) PASS in both run dirs (`status.txt`); counts TASK OK on all 26 units of each (`ruler/*.check`).
- F: b38 (A1 units 1.11, 1.8) and q38 (A2 units 2.11, 2.8) PASS: 20 dense outputs each, 0 with `<think>`, 0 empty or blank.
- D: task-0 dense 100 for both. Gates F and D were read late (`--late F --late D`); nothing stopped, and the recomputed gates decide.
- Row 2n (amendment 1): 0 degenerate rows for each model. Row 3: dense passed every floor.

## Results

| model | config | mean | Δ all | pass 1 Δ | pass 2 Δ |
|---|---|---|---|---|---|
| b38 | `quest:27648:f3` | 95.38 | −0.38 [−1.15, 0.00] | −0.77 [−2.31, 0.00] | 0.00 [0.00, 0.00] |
| b38 | `pf:0.1:512:d512` | 94.56 | −1.21 [−2.17, −0.58] | −1.77 [−3.50, −0.62] | −0.65 [−1.12, −0.31] |
| b38 | **combined** | 94.17 | **−1.60 [−3.25, −0.15]** | −2.38 [−5.58, +0.46] | −0.81 [−1.31, −0.38] |
| q38 | `quest:27648:f3` | 95.38 | −0.38 [−1.15, 0.00] | −0.77 [−2.31, 0.00] | 0.00 [0.00, 0.00] |
| q38 | `pf:0.1:512:d512` | 94.50 | −1.27 [−2.25, −0.60] | −1.77 [−3.54, −0.58] | −0.77 [−1.38, −0.31] |
| q38 | **combined** | 94.12 | **−1.65 [−3.25, −0.17]** | −1.85 [−4.65, +0.81] | −1.46 [−3.08, −0.54] |

- Combined per task (`per_task_delta`): b38 cwe −7, niah_multikey_3 −5, qa_2 −5, niah_multivalue −3.75; q38 cwe −9, qa_1 −5, qa_2 −5, niah_multivalue −2.5. Every other task 0; vt 100 in every config of both models.
- Dense scores 100 on 11 of 13 tasks at both precisions (qa_1 85, qa_2 60): on those tasks only a loss can be measured.
- Output kinds: all 2,080 outputs are direct answers; no `<think>` text, no empty or blank output.
- b38 and q38 dense rows are token-identical on 210 of 260 prompts (`dense_token_identity_b38_q38`).
- Gather / dense prefill time (informative): median 0.675 (b38) and 0.682 (q38).

## Predictions

| # | prediction | outcome |
|---|---|---|
| P1 | gates pass on both boxes; counts hold; both cohorts exact | **partly held**: box B did not complete its list and was excluded; all gates passed in A1 and A2 and both cohorts are exact only through amendment 1's rerun |
| P2 | ≤ 5 empty or blank outputs per config per model; no `<think>` text | held: 0 and 0 |
| P3 | dense means q38 and b38 in 86–94 | **missed**: both 95.77, above the range |
| P3 | qa_1 dense ≥ 50 for both | held: 85 and 85 |
| P4 | q38 combined Δ in −4.0 to +0.5 (point −2.0); PASS | held: −1.65; PASS |
| P5 | b38 combined Δ in −5.0 to +0.5 (point −2.5); PASS | held: −1.60; PASS |
| P6 | largest combined single-task loss is cwe in both models | held: b38 cwe −7, q38 cwe −9; vt 0 in both |
| P7 | BOTH_PASS (p ≈ 0.35) | held |

## Deviations

1. Box B excluded and its list rerun on box A (amendment 1, after data existed). Because of the rerun, gates F and D for q38 were read about 9 h after launch.
2. Run A2 was launched by an on-box script 91 s after A1's COMPLETE line and before A1 was pulled (the amendment said "after A1 ends and is pulled"); A1's files were not touched and were pulled and checked later.
3. A side test (q38 dense and pf on 10 prompts, not PF-NT data) shared the GPU with b38 units 1.2 and 1.3 in A1. Those units slowed (1,511 s and 1,787 s against 999–1,173 s for the other pass-1 b38 units), which also explains the gather / dense outliers at ids 2006 (1.451) and 3006 (0.410). The ratio is informative only; all scores on ids 2006–3006 are 100 in every config, but those units were not rerun on their own.

## What this does not settle

The emulation / gather path only, not a dedicated kernel; no speed result (the prefill ratio is informative); one workstation card, one driver, toolkit and power limit; 20 prompts per task with most tasks at ceiling; thinking-on use, q8_0 KV and other generation lengths; other models and weight formats.

## Files

- `vastai38/<run>/`: `status.txt`, `ruler/<model>_p<pass>_t<task>.jsonl` (`llama-sparse-tail --gen` rows) with their `.check` files and the per-model cohort `.json`, `nsys/` (gate U kernel summaries `*_cuda_gpu_kern_sum.csv` and rows), `commit.txt`, `sha.txt`, `gpu.txt`, `cmake_fork.settings`, `tbo_*.txt` (gate K outputs).
- `analysis/manifest.json`, `analysis/decision.json`; `inputs/pf_nt_q38.i32.meta.json` (task and references per prompt id, used for scoring).
- Not committed: the `ruler/*.err` llama.cpp logs (23 MB per run dir; `pfn_decide.py` does not read them), the nsys reports (`.nsys-rep`, `.sqlite`, 210 MB) and nsys logs, `gpu_log.csv`, build logs, `pgids`, the prompt token file `pf_nt_q38.i32` (129 MB), box B's run dir and the side tests.
