# PORT-MQ4 verdict (Qwen3-30B-A3B-Instruct-2507 Q4_K_M, KV q8_0, RTX 5090 32 GB, release build e7059e52a)

Box: RTX 5090 (sm 12.0, driver 580.178.04), Ryzen 7 7800X3D. Model SHA-256 6c997b8a… matches the protocol (`sha.txt`); the RULER token file is byte-identical to PORT-MQ's and PORT-RQ's (`tokens_sha.txt`).

Smoke: quest:12288 / dense at 65536 = 1.192 (≥ 1.02, continue). Prediction (smoke passes) held: the fork builds and runs on Blackwell.

Speed (llama-bench tg64; 10 reps ≤ 32k, 5 above):

| depth | budget | dense tok/s | quest tok/s | speedup |
|---|---|---|---|---|
| 8192 | 2048 | 244.12 | 161.92 | 0.66x |
| 16384 | 3072 | 198.14 | 147.32 | 0.74x |
| 32768 | 6144 | 138.14 | 128.99 | 0.93x |
| 65536 | 12288 | 83.77 | 103.72 | 1.24x |
| 131072 | 27648 | 46.05 | 79.17 | **1.72x** |

- Prediction S (1.3–1.9x at 131072, point 1.55) **held**: 1.72x, above the point estimate. Below 1x at ≤ 32k, as on every other box.

RULER at 128k, 13 tasks x 20 prompts, quest:27648 (`mq_score.json`, `port_r_score.py --model Qwen/Qwen3-30B-A3B-Instruct-2507`):

| | dense | quest | Δ [95% CI] |
|---|---|---|---|
| mean | 87.8 | 86.8 | −1.0 [−2.2, −0.02] |
| single needle (3) | 100.0 | 100.0 | 0.0 |
| multi-needle (5) | 100.0 | 99.75 | −0.25 (multiquery −1.25) |
| other (5) | 68.2 | 65.9 | −2.2 (qa_1 −5, qa_2 −5, fwe −1.7, cwe +0.5) |

- Gate **PASS** (Δ ≥ −3 and lower bound ≥ −6). Prediction Q (PASS, Δ in [−3, +1]) held.
- The upper CI bound is just below zero (−0.02): unlike PORT-MQ, the small loss is distinguishable from zero at this sample size. It sits in qa_1 and qa_2 (one prompt each of 20) and fwe; the needle tasks are intact.
- Dense at Q4_K_M (87.8) is at the Q8_0 level of PORT-MQ (87.2).
- No deviations from the protocol. `mq_gen.err` (7.5 MB of llama.cpp logs) is not committed.
- Consequence (pre-registered): the Reddit draft adds the Q4_K_M / RTX 5090 row; the bench README gets it next to PORT-MQ.
