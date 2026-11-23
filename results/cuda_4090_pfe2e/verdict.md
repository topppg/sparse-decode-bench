# PF-E2E arm D verdict (RTX 4090 24 GB at 330 W, speed at 128k; Qwen3.8-27B UD-Q4_K_M, KV q8_0)

Protocol `protocols/pf-e2e.md` and amendments 2 and 3; the full verdict is `results/pfe2e_analysis/verdict.md`, the decision `results/pfe2e_analysis/decision.json` (`speed_D`). These are the on-box files of box vastai51, run 20261121T225917Z, the third launch on that box and the only one that is arm D's data.

Box: RTX 4090 (sm 8.9, driver 580.173.02), **330 W power limit** (stock 450 W; recorded, not gated), so absolute times are slower than a stock card's. Fork tree 9e3e51c75 and upstream c6a25fc62 from on-box git mirrors of the first launch's verified clones (amendment 3); model SHA-256 checked. Every setup gate passed, gate B PASS at 128k, final line COMPLETE, 17/17 rows OK.

| block | best dense | best dense wall | `pfsd27648` wall | ratio |
|---|---|---|---|---|
| 1 | `fork_mtp` | 102.39 s | 77.02 s | 1.3293 |
| 2 | `up_mtp` | 102.55 s | 77.16 s | **1.3291** (quoted) |

- Quoted R_128 = **1.329** (`pfsd55296` 1.291, `pf_only` 1.195). Prediction P10 (1.10–1.35, point 1.22) held.
- Registered stopping rule for sparse prefill on this card: `fork_dense` TTFT / `pfsd27648` TTFT = 1.398 ≥ 1.30, cleared.
- Fork and upstream are level at dense prefill here (TTFT 75.5–75.9 s), unlike on the RTX 5090.
- Block drift at most 0.63%; every config's four timed outputs identical.
- Quality basis (registered sentence, gate B PASS on this card): quality rests on the bit identity of GATHER=2 with GATHER=1 (gate B) plus PF-NT and PF-CONF at 128k; no RULER was run on this card.

The two earlier launches on this box are not copied: 20261121T221741Z ended `FAILED sha q38` (the model download stopped at 90%; builds and gate K had passed) and 20261121T225103Z ended `FAILED clone` (network failure); neither ran gate B, a reference or a timed row (amendments 2 and 3).

Files: as `cuda_5090_pfe2e/` (128k only). Not committed: `tbo_fa_all.txt` (1.4 MB; its PASS line is in `status.txt`), `bit/tail_*.err` (gate B llama.cpp logs), build and cmake logs, `pgids`.
