<!-- first committed in the project repository as 5932f27 at 2026-11-06T14:20:30+01:00 -->

## PORT-R3 (2026-11-06): full RULER with the quest score at the KL budget (pre-registered after PORT-D D-A; before any R3 data)

**Question.** D-A showed that on the six tasks that lost most in R1, quest at 27648 cells is −3.4 against dense, where mv is −15.7. Does quest pass the R1/R4 RULER gate on all 13 tasks at 27648 cells, the budget of the 128k speed table?

**Setup.** As R1 and R4: `vastai11` (RTX 4090 24 GB), llama.cpp `sparse-cuda` 8213b023d, Llama-3.1-8B q8_0 (SHA-256 checked against R1), KV q8_0, page stats f16, the 260 R1 prompts (`ruler8b.i32`, SHA-256 checked against R1), scoring `port_r_score.py`. Configs: dense, quest:27648 (one process). Speed first: `llama-bench -d 131072 -n 64 -r 3`, dense, mv:27648 and quest:27648.

**Gate (as R1/R4).** Δ = quest − dense ≥ −3.0 points on the 13-task mean, and the 95% paired bootstrap lower bound ≥ −6.0.

**Checks (no gate).** Dense outputs against R1 dense, token by token. quest on the 120 D-A prompts against D-A quest (different box), score and token match reported.

**Predictions.** PASS, with Δ within [−3.0, −0.5]: the six D-A tasks contribute about 6/13 × −3.4 ≈ −1.6, and the other seven were within 5 points each under mv in R1. quest decode speed at 27648 within 3% of mv at 27648.

**Consequence (fixed now).** PASS: the released default is quest, and the 128k speed table at 27648/28672 cells stands as the operating point with RULER measured on it. FAIL: the released default is quest at B\* = 49152, with quality from PORT-L L2 (UltraLong-8B, 128k and 256k); the 27648 point is reported as the faster end of the dial with its R3 RULER score. No further RULER runs either way.
