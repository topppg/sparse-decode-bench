# PF-E2E RULER verdict (two RTX 5090 32 GB, RULER at 256k, 13 tasks × 20; Qwen3.8-27B UD-Q4_K_M, KV q8_0, chat format with thinking off)

Protocol `protocols/pf-e2e.md` and amendment 4 (written after the data); the full verdict is `results/pfe2e_analysis/verdict.md`, the decisions `results/pfe2e_analysis/decision.json` (with amendment 4) and `decision_as_registered.json` (without). On-box files of two boxes:
- `vastai49/`: box B, pass 1 (prompts 0–9), run 20261121T222607Z, final line `FAILED pass 1 task 12 (rc 2, see status)`. Its last unit is in `vastai49/tmp/`, where the box left the unit it rejected.
- `vastai50/`: box C, pass 2 (prompts 10–19), run 20261121T222610Z, COMPLETE, 13 units, 650 rows.

Both: RTX 5090, 500 W, driver 580.178.04, 30 GiB RAM (amendment 1); fork tree 9e3e51c75; model SHA-256 checked; gate K and gate A PASS (`status.txt`).

**Outcome.** As registered: quality **INVALID → NO_DECISION** (box B has no `Q38 UNITS COMPLETE` line), consequence SPEED_ONLY. With amendment 4: **PASS_27648**.

| config | Δ vs dense | 95% CI | gate (Δ ≥ −3, lower ≥ −6) |
|---|---|---|---|
| `pf:0.1:512:d512+quest:27648:f3` (primary) | **−2.40** | [−4.06, −0.88] | pass (amendment 4) |
| `pf:0.1:512:d512+quest:55296:f3` | −1.88 | [−3.37, −0.50] | pass |
| `pf:0.1:512:d512` | −1.73 | [−3.04, −0.73] | pass |
| `quest:27648:f3` | −1.73 | [−3.27, −0.46] | pass |

- Dense 13-task mean 95.26; 100 on 10 tasks (fwe 98.3, qa_1 85.0, qa_2 55.0). Prediction P11's dense range (70–92) missed (above); its Δ range (−5 to +1) held.
- Primary per task: cwe −16.5, qa_1 −5, qa_2 −5, niah_multivalue −3.75, vt −1; the other eight tasks 0.
- Pass 1 alone −2.96 [−6.04, −0.04], pass 2 alone −1.85 [−2.77, −1.15] (descriptive). Without qa_1 −2.19 [−3.33, −1.33] (descriptive).
- 0 empty outputs, 0 degenerate rows, 0 `<think>` text in 1,300 outputs; gates F and D PASS on both boxes.

**Why amendment 4.** Prompt 12004 is 256,002 tokens long; its prefill of 256,001 tokens ends in a one-token ubatch, which the engine runs dense and does not count. The box's count checker expected it to be counted (16 × 512 dense rows against the engine's 16 × 511), marked unit q38 1.12 invalid (`vastai49/tmp/q38_p1_t12.check`) and ended the run. Amendment 4 corrects the checker (`scripts/pf_e2e/box/in/pfe_check.py`, two lines) and `e2e_decide.py --amend4` re-checks that unit from `tmp/` with every registered unit rule: `TASK OK`. The run is deterministic, so no GPU rerun was made. The score thresholds did not change.

Files per box: `status.txt`, `ruler/q38_p<pass>_t<task>.jsonl` (`llama-sparse-tail --gen` rows: id, config, output token ids, sparse-prefill counts, timings), `.check` (the box's unit check), `.err.gz` (llama.cpp logs, gzipped from the on-box `.err`; the decision reads the conversion-count and graph-reuse lines), `gate_a/rows.jsonl`, `vastai50/ruler/cohort_q38.json`, `commit.txt`, `sha.txt`, `gpu.txt`, `cmake_fork.settings`, `tbo_*.txt`. Scoring needs `results/pfe2e_analysis/inputs/pfe_ruler256_q38.i32.meta.json` (task and references per prompt id).

Not committed: `tbo_fa_all.txt` (1.4 MB each; PASS lines in `status.txt`), `gate_a/err.txt` (0.35 MB each), `gpu_log.csv`, build and cmake logs, `pgids`, operator notes.
