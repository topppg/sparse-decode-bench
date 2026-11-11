# PORT-MQ verdict (Qwen3-30B-A3B-Instruct-2507 Q8_0, KV q8_0, RTX 4090 48 GB, release build e7059e52a)

Smoke: quest:12288 / dense at 65536 = 1.119 (≥ 1.02, continue).

Speed (llama-bench tg64; 10 reps ≤ 32k, 5 above):

| depth | budget | dense tok/s | quest tok/s | speedup |
|---|---|---|---|---|
| 8192 | 2048 | 148.03 | 104.90 | 0.71x |
| 16384 | 3072 | 122.63 | 99.93 | 0.81x |
| 32768 | 6144 | 91.60 | 89.14 | 0.97x |
| 65536 | 12288 | 61.18 | 72.93 | 1.19x |
| 131072 | 27648 | 37.38 | 52.63 | **1.41x** |

- Prediction S (1.5–2.3x at 131072, point 1.9) **missed**: 1.41x. Not tested, a likely reading: the band was implicitly calibrated on the RTX 3090 rows; on a 4090 48 GB, Llama-3.1-8B with mv gave 1.31x (PORT-C), so the MoE does gain more than the dense 8B on this GPU class, but less than predicted.

RULER at 128k, 13 tasks x 20 prompts, quest:27648:

| | dense | quest | Δ [95% CI] |
|---|---|---|---|
| mean | 87.2 | 86.6 | −0.6 [−1.9, +0.4] |
| single needle (3) | 100.0 | 100.0 | 0.0 |
| multi-needle (5) | 98.0 | 97.0 | −1.0 (multikey_3 −5, one prompt) |
| other (5) | 68.8 | 68.1 | −0.7 (qa_2 −5, fwe +1.7) |

- Gate **PASS** (Δ ≥ −3 and lower bound ≥ −6). Prediction Q (PASS, Δ in [−3, +1]) held.
- The token file is byte-identical to PORT-RQ's `rulerq.i32` (same Qwen3 tokenizer), so Qwen3-8B and this model saw the same prompts.
- Deviation: the first attempt failed to download `tokenizer.json` from the hub on the box (`status_attempt1.txt`, `tokens_attempt1.log`); the same revision's file (SHA-256 aeb13307…, equal to the hub's LFS oid) was copied in and the run restarted. Model and engine unchanged.
- `mq_gen.err` (7.5 MB of llama.cpp logs) is not committed.
