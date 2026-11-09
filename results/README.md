# Results

Raw outputs and analyses, one folder per pre-registered experiment (protocol in `protocols/`). Folder names keep the protocol labels used in the protocols, so references there stay valid.

| folder | protocol | hardware | contents | used in README |
|---|---|---|---|---|
| `cuda_4090_portc` | PORT-C, C2, C3 | RTX 4090 48 GB | first CUDA port, kernel fixes (`c3fix*`), 8B KL and needles | Speed (4090 rows), What did not work |
| `cuda_3090_portg` | PORT-G | RTX 3090 | dense/sparse speed 8k–128k, microbenchmarks, cost-model predictions, Qwen3 KL and needles | Speed (3090 rows), cost model, cheaper checks |
| `cuda_3090_portr` | PORT-R | RTX 3090 | speed by budget, wall-clock with prefill | Dial, Wall-clock |
| `cuda_4090_portr` | PORT-R | RTX 4090 24 GB | RULER with mv (R1), the RULER prompt metadata, speed by budget | Quality (mv), Why mv failed |
| `cuda_4090_portr2` | PORT-R2 | RTX 4090 24 GB | RULER with mv at 49152 and 65536 cells; re-validation of the two upstream kernel PRs | Why mv failed (larger budgets), What did not work |
| `cuda_4090_portr3` | PORT-R3 | RTX 4090 24 GB | RULER with quest on Llama-3.1-8B; quest vs mv speed | Quality, Speed (quest vs mv) |
| `cuda_4090_portd` | PORT-D | RTX 4090 48 GB and 24 GB | the six losing tasks under single changes (score, dense layers, f32 statistics) | Why mv failed |
| `cuda_4090_48gb_portl` | PORT-L | RTX 4090 48 GB | UltraLong-8B speed, KL and RULER at 128k and 256k; cost model at 256k | Longer context, cost model |
| `cuda_4090_portrq` | PORT-RQ | RTX 4090 24 GB | RULER with quest on Qwen3-8B and Llama-3.2-3B; post-hoc multikey analysis | Quality |
| `cuda_4090_48gb_featcheck` | (validation) | RTX 4090 48 GB | released branch vs research build on 30 RULER prompts, repeated | Scope and limits (reproducibility) |
| `pageq_reference` | PAGE-Q | Python reference (fp32 with TF32) | KL vs dense by budget and page score on 4 PG19 books, Llama-3.2-3B and Llama-3.1-8B | What did not work (in-engine KL vs reference) |
| `metal_port_k0` … `metal_port_k3`, `metal_port_k2_v2` | PORT-K0–K3 | Apple M4 Pro | Metal kernel microbenchmarks | Try it (Metal note) |
| `metal_port_e1` | PORT-E1, E2 | Apple silicon (Metal) | in-engine page selection and cached statistics are exact | — |

File types:
- `*_gen.jsonl`: one line per prompt and config from `llama-sparse-tail --gen`: `id`, `config` (`dense`, `quest:<B>`, `mv:<B>`), output token ids.
- `*.i32.meta.json`: per prompt id, the RULER task and reference answers.
- `*scores.json`, `*_score.json`: RULER scores from `scripts/ruler_score.py`.
- `bench/*.jsonl`: `llama-bench -o jsonl` output.
- `micro/*.txt`: `test-backend-ops perf` output.
- `commit.txt`, `sha.txt`, `gpu.txt`, `env.txt`, `status.txt`: build commit, model hashes, hardware, run log.
