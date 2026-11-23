# Results

Raw outputs and analyses, one folder per pre-registered experiment (protocol in `protocols/`). Folder names keep the protocol labels used in the protocols, so references there stay valid.

| folder | protocol | hardware | contents | used in README |
|---|---|---|---|---|
| `cuda_4090_portc` | PORT-C, C2, C3 | RTX 4090 48 GB | first CUDA port, kernel fixes (`c3fix*`), 8B KL and needles | Speed (4090 rows), What did not work |
| `cuda_3090_portg` | PORT-G | RTX 3090 (300 W) | dense/sparse speed 8k–128k, microbenchmarks, cost-model predictions, Qwen3 KL and needles | Speed (3090 rows), cost model, cheaper checks |
| `cuda_3090_portr` | PORT-R | RTX 3090 (300 W) | speed by budget, wall-clock with prefill | Dial, Wall-clock |
| `cuda_4090_portr` | PORT-R | RTX 4090 24 GB | RULER with mv (R1), the RULER prompt metadata, speed by budget | Quality (mv), Why mv failed |
| `cuda_4090_portr2` | PORT-R2 | RTX 4090 24 GB | RULER with mv at 49152 and 65536 cells; re-validation of the two upstream kernel PRs | Why mv failed (larger budgets), What did not work |
| `cuda_4090_portr3` | PORT-R3 | RTX 4090 24 GB | RULER with quest on Llama-3.1-8B; quest vs mv speed | Quality, Speed (quest vs mv) |
| `cuda_4090_portd` | PORT-D | RTX 4090 48 GB and 24 GB | the six losing tasks under single changes (score, dense layers, f32 statistics) | Why mv failed |
| `cuda_4090_48gb_portl` | PORT-L | RTX 4090 48 GB | UltraLong-8B speed, KL and RULER at 128k and 256k; cost model at 256k | Longer context, cost model |
| `cuda_4090_portrq` | PORT-RQ | RTX 4090 24 GB | RULER with quest on Qwen3-8B and Llama-3.2-3B; post-hoc multikey analysis | Quality |
| `cuda_3090_portgq` | PORT-GQ | RTX 3090 (a second box, 350 W) | dense, mv and quest speed 8k–128k on the release build | Speed |
| `cuda_4090_48gb_portmq` | PORT-MQ | RTX 4090 48 GB | Qwen3-30B-A3B (MoE): smoke test, speed 8k–128k, RULER with quest | Speed, Quality |
| `cuda_5090_portmq4` | PORT-MQ4 | RTX 5090 32 GB | Qwen3-30B-A3B (MoE) at Q4_K_M: smoke test, speed 8k–128k, RULER with quest | Speed, Quality |
| `cuda_6000_pfnt` | PF-NT | RTX PRO 6000 Blackwell 96 GB | RULER at 128k for sparse prefill and sparse decode, Qwen3.8-27B BF16 and UD-Q4_K_M, f16 KV, chat format; decision | Sparse prefill (quality at 128k) |
| `cuda_5090_pfe2e` | PF-E2E arm A | RTX 5090 32 GB (500 W) | one request end to end at 256k and 128k through `llama-server`, sparse prefill + sparse decode against five dense setups | Sparse prefill (end to end) |
| `cuda_4090_pfe2e` | PF-E2E arm D | RTX 4090 24 GB (330 W) | the same at 128k, speed only | Sparse prefill (end to end) |
| `cuda_5090_pfe2e_ruler` | PF-E2E boxes B and C | two RTX 5090 32 GB | RULER at 256k, 13 tasks × 20, two passes (`vastai49/`, `vastai50/`), the unit recovered by amendment 4 in `vastai49/tmp/` | Sparse prefill (quality at 256k) |
| `pfe2e_analysis` | PF-E2E | (analysis) | the decision with and without amendment 4, the RULER manifest, prompt metadata, the full verdict | Sparse prefill |
| `cuda_5090_pfrebase_dev` | (development run, not pre-registered) | RTX 5090 32 GB (500 W) | the sparse path rebased onto upstream c6a25fc62, one block of the PF-E2E rows | Sparse prefill (after the rebase) |
| `cuda_4090_48gb_featcheck` | (validation) | RTX 4090 48 GB | released branch vs research build on 30 RULER prompts, repeated | Scope and limits (reproducibility) |
| `pageq_reference` | PAGE-Q | Python reference (fp32 with TF32) | KL vs dense by budget and page score on 4 PG19 books, Llama-3.2-3B and Llama-3.1-8B | What did not work (in-engine KL vs reference) |
| `metal_port_k0` … `metal_port_k3`, `metal_port_k2_v2` | PORT-K0–K3 | Apple M4 Pro | Metal kernel microbenchmarks | Try it (Metal note) |
| `metal_port_e1` | PORT-E1, E2 | Apple silicon (Metal) | in-engine page selection matches the reference (passed); cached page statistics are exact in f32 but not bit-identical in f16 (max logit difference 0.074, mean KL 1.6e-8), so the f16 exactness gate failed | — |

File types:
- `*_gen.jsonl`: one line per prompt and config from `llama-sparse-tail --gen`: `id`, `config` (`dense`, `quest:<B>`, `mv:<B>`), output token ids.
- `*.i32.meta.json`: per prompt id, the RULER task and reference answers.
- `*scores.json`, `*_score.json`: RULER scores from `scripts/ruler_score.py`.
- `bench/*.jsonl`: `llama-bench -o jsonl` output.
- `micro/*.txt`: `test-backend-ops perf` output.
- `commit.txt`, `sha.txt`, `gpu.txt`, `env.txt`, `status.txt`: build commit, model hashes, hardware, run log.
- Sparse prefill folders (`*_pfnt`, `*_pfe2e*`, `*_pfrebase_dev`): `e2e/<len>_b<block>_<config>.jsonl` are `llama-server` client rows (timed requests with output ids and timings), `e2e/srv_*.log` the server logs; `ruler/<model>_p<pass>_t<task>.jsonl` are `llama-sparse-tail --gen` rows with `.check` (the box's unit check); `*.err.gz` are llama.cpp stderr logs, gzipped from the on-box `*.err` (the decision reads them; `scripts/pf_e2e/decide.sh` decompresses them). Each folder's `verdict.md` lists what was not committed.
