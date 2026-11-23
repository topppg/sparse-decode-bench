# PF-E2E arm A verdict (RTX 5090 32 GB, speed at 256k and 128k; Qwen3.8-27B UD-Q4_K_M, KV q8_0)

Protocol `protocols/pf-e2e.md`; the full verdict, with the quality result and the predictions, is `results/pfe2e_analysis/verdict.md`, and the decision `results/pfe2e_analysis/decision.json` (`speed_A`). These are the on-box files of box vastai48, run 20261121T222603Z.

Box: RTX 5090 (sm 12.0, driver 580.178.04, CUDA 13.0), 500 W power limit (the card reports a 600 W default limit; recorded, not gated), Ryzen 7 7800X3D, 30 GiB RAM (amendment 1). Fork tree 9e3e51c75 (f7d80009d + 15 patches) and upstream c6a25fc62 built on the box; model SHA-256 checked (`sha.txt`). Every setup gate passed (`status.txt`), gate B PASS at both lengths (`bit/bin_sha256.txt`), final line COMPLETE, 37/37 rows OK.

One request: book prompt (259,525 or 129,526 tokens) + 1,000 generated tokens through `llama-server`, two timed requests per row, two blocks in reverse order. Median wall of the block that gives the quoted ratio (the lower block):

| length | best dense | best dense wall | `pfsd27648` wall | R | `pfsd55296` | `pf_only` | vs `fork_dense` |
|---|---|---|---|---|---|---|---|
| 256k | `up_dense` | 172.45 s | 107.01 s | **1.612** | 1.589 | 1.424 | 2.31 (246.72 s) |
| 128k | `up_mtp` | 67.31 s | 52.95 s | **1.271** | 1.233 | 1.140 | 1.65 (87.23 s) |

- Length effect R_256 − R_128 = +0.340: GROWS (registered margin 0.3).
- 256k: TTFT 145.36 s (`up_dense`) against 92.28 s; decode 36.99 against 68.20 tok/s. `fork_dense` TTFT 218.40 s.
- Block drift at most 0.12%; every config's four timed outputs identical; gate N PASS.
- Peak memory.used at 256k (`gpu_log.csv` and `stages.txt`): `up_dense` 25,604 MiB (25.0 GiB), `fork_dense_ub2048` 26,694 MiB (26.1 GiB), `up_mtp` 28,362 MiB (27.7 GiB; `fork_mtp` 28,364), sparse 27,292 MiB (26.7 GiB).
- Predictions P5 (R_256 1.7–2.6, point 2.2) and P6 (R_128 1.35–1.80, point 1.55) **missed**: the best dense was upstream, whose dense prefill on this card is much faster than the fork base's (256k TTFT 145.4 s against 218.4 s).

Files: `status.txt` (timeline and every gate line), `e2e/<len>_b<block>_<config>.jsonl` (client rows: smoke and timed requests, output token ids, timings), `e2e/srv_*.log` (server logs: conversion counts, OOM check), `e2e/client.log`, `ref/gen_<len>.jsonl` (`llama-sparse-tail` references) and `ref/gen_<len>.err.gz` (their llama.cpp logs, gzipped; the decision reads the conversion count from them), `bit/` (gate B rows and logit SHA-256s), `matrix.txt`, `commit*.txt`, `sha.txt`, `gpu.txt`, `gpu_log.csv`, `stages.txt`, `cmake_*.settings`, `tbo_*.txt` (gate K outputs).

Not committed: `tbo_fa_all.txt` (1.4 MB, the every-`FLASH_ATTN_EXT` gate K output; its PASS line is in `status.txt`), `bit/tail_*.err` (gate B llama.cpp logs, 0.4–0.5 MB each), build and cmake logs, `pgids`, the empty `dl_q38.log`.
