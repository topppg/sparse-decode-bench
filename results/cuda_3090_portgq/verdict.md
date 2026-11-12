# PORT-GQ verdict (RTX 3090, release build e7059e52a)

Speedup vs dense at 131072, same box (llama-bench tg64, 5 reps):

| model | dense | mv | quest | quest/mv | gate (quest ≥ 0.95 × mv) | mv vs PORT-G |
|---|---|---|---|---|---|---|
| Llama-3.2-3B (f16, 28672) | 44.44 | 61.11 (1.375x) | 61.32 (1.380x) | 1.003 | PASS | +1.9% |
| Llama-3.1-8B (q8_0, 27648) | 23.67 | 38.96 (1.646x) | 38.85 (1.641x) | 0.997 | PASS | −7.0% |
| Qwen3-8B-Base YaRN (q8_0, 27648) | 21.46 | 36.20 (1.687x) | 36.08 (1.681x) | 0.997 | PASS | −6.8% |

- Prediction (quest within ±3% of mv) held: quest/mv is 0.990–1.003 at every depth for all three models.
- Check: this 3090 runs dense faster than the PORT-G box (8B 23.67 vs 21.86 tok/s, Qwen 21.46 vs 19.90); sparse is about the same (38.85 vs 38.75, 36.08 vs 35.97). This box has a 350 W power limit and CUDA 12.8; the PORT-G box had 300 W and CUDA 13.4. The 8B mv speedup differs from PORT-G by 7.0%, just at the reporting threshold: box-to-box variation.
- Consequence (pre-registered): the Reddit speed table and chart use quest from this run, so the 3090 headline is 1.64x / 1.68x / 1.38x instead of PORT-G's mv 1.77x / 1.81x / 1.35x.
- Model SHA-256s match PORT-G (`sha_check.txt`).
