# PF-E2E verdict (Qwen3.8-27B UD-Q4_K_M, KV q8_0; RTX 5090 and RTX 4090 speed, RULER at 256k on two RTX 5090s)

Protocol: `protocols/pf-e2e.md` and amendments 1–4. Decision: `decision.json` (`e2e_decide.py --amend4`); as registered, without amendment 4: `decision_as_registered.json`. Both are recomputed byte for byte from the committed results by `scripts/pf_e2e/decide.sh` (see `scripts/README.md`). Adapted from the project repository's verdict, with its erratum (2026-11-23) applied in the text: the RTX 4090 times and the peak-memory units below are the corrected ones.

**Outcome: PASS_27648_MODERATE, resting on amendment 4 (written after the data).** As registered, without amendment 4, the quality decision is INVALID → NO_DECISION and the consequence row is SPEED_ONLY: box B's run ended `FAILED pass 1 task 12` because the count checker mis-counted one prompt (below). The speed results do not depend on amendment 4.

## Runs

| box | arm | card | run | folder | end |
|---|---|---|---|---|---|
| vastai48 | A, speed 256k + 128k | RTX 5090 32 GB, 500 W, 30 GiB RAM (amendment 1) | 20261121T222603Z | `cuda_5090_pfe2e/` | COMPLETE, 37/37 rows OK |
| vastai51 | D, speed 128k | RTX 4090 24 GB, **330 W** (recorded, not gated) | 20261121T225917Z (third launch; amendments 2, 3) | `cuda_4090_pfe2e/` | COMPLETE, 17/17 rows OK |
| vastai49 | B, RULER pass 1 | RTX 5090, 500 W, 30 GiB | 20261121T222607Z | `cuda_5090_pfe2e_ruler/vastai49/` | FAILED pass 1 task 12 (count check, amendment 4) |
| vastai50 | C, RULER pass 2 | RTX 5090, 500 W, 30 GiB | 20261121T222610Z | `cuda_5090_pfe2e_ruler/vastai50/` | COMPLETE, 13 units, 650/650 rows |

Model Qwen3.8-27B UD-Q4_K_M (SHA-256 1bce504d…), q8_0 KV, fork f7d80009d + patches 0001–0007 + page-list patches 0001–0008 (tree 9e3e51c75), upstream c6a25fc62. Book prompt 259,525 / 129,526 tokens, 1,000 generated tokens; RULER 256k in the chat format, 13 tasks × 20.

The two earlier launches of arm D on vastai51 (20261121T221741Z, `FAILED sha q38` after a model download failure; 20261121T225103Z, `FAILED clone` after a network failure) ran no gate B, reference or timed row. They are the setup failures of amendments 2 and 3 and are not copied here.

## Speed (registered, unaffected by amendment 4)

Quoted ratio R = best dense median wall / config median wall, the lower of two blocks. Times are the medians of the block that gives R.

| card, length | best dense | best dense wall | `pfsd27648` wall | **R** | `pfsd55296` | `pf_only` | TTFT ratio vs best / vs `fork_dense` |
|---|---|---|---|---|---|---|---|
| 5090, 256k | `up_dense` (both blocks) | 172.45 s | 107.01 s | **1.612** | 1.589 | 1.424 | 1.575 / 2.367 |
| 5090, 128k | `up_mtp` (both blocks) | 67.31 s | 52.95 s | **1.271** | 1.233 | 1.140 | 1.408 / 1.743 |
| 4090, 128k | `up_mtp` (block 2; `fork_mtp` 102.39 s in block 1) | 102.55 s | 77.16 s | **1.329** | 1.291 | 1.195 | 1.499 / 1.398 |

- On the 4090, block 1 gives 102.39 s (`fork_mtp`) against 77.02 s, R 1.329 as well (1.3293 against block 2's 1.3291).
- Gate B (GATHER=2 logits byte-identical to GATHER=1, 32 steps) PASS at 256k and 128k on the 5090 and at 128k on the 4090; gate N PASS; every config's four timed outputs identical. Block drift: at most 0.12% on the 5090 and 0.63% on the 4090 (limit ±3%).
- Length effect (`pfsd27648`): R_256 − R_128 = +0.340 → **GROWS** (registered margin 0.3).
- RTX 4090 stopping rule (registered): `fork_dense` TTFT / `pfsd27648` TTFT = 1.398 ≥ 1.30 → cleared.
- Against the fork's own dense (`fork_dense`, informative): 2.31× end to end at 256k on the 5090 (246.7 s).
- Read during the run (not a registered comparison): upstream c6a25fc62 is much faster than the fork base at dense prefill on the 5090 (256k TTFT 145.4 s against 218.4 s; 128k 51.1 s against 67.1 s); on the 4090 at 128k they are level (75.5–75.9 s). The best dense is therefore upstream on the 5090, and the quoted ratios are against it.
- 256k decode on the 5090: `up_dense` 37.0 tok/s, `pfsd27648` 68.2 tok/s; TTFT 145.4 s against 92.3 s.
- Peak memory.used (5090, 256k): `up_dense` 25,604 MiB (25.0 GiB), `fork_dense_ub2048` 26,694 MiB (26.1 GiB), MTP 28,362 MiB (27.7 GiB), sparse 27,292 MiB (26.7 GiB).

## Quality at 256k (with amendment 4)

Δ = config − dense, points, 260 pairs, 95% bootstrap CI; gate Δ ≥ −3 and lower bound ≥ −6.

| config | Δ | 95% CI | gate |
|---|---|---|---|
| `pf:0.1:512:d512+quest:27648:f3` (primary) | **−2.40** | [−4.06, −0.88] | **pass** |
| `pf:0.1:512:d512+quest:55296:f3` | −1.88 | [−3.37, −0.50] | pass |
| `pf:0.1:512:d512` | −1.73 | [−3.04, −0.73] | pass |
| `quest:27648:f3` | −1.73 | [−3.27, −0.46] | pass |

- Only the primary row decides (rows 5–7 of the decision are fallbacks).
- Dense 13-task mean 95.26 (100 on 10 tasks; fwe 98.3, qa_1 85.0, qa_2 55.0). 0 empty outputs, 0 degenerate rows, 0 `<think>` text. Gates F and D PASS on box B (registered) and C (descriptive). Sparse/dense prefill time median 0.411.
- Primary per task: cwe −16.5, qa_1 −5.0, qa_2 −5.0, niah_multivalue −3.75, vt −1.0, all others 0.
- Per pass (descriptive): pass 1 −2.96 [−6.04, −0.04], pass 2 −1.85 [−2.77, −1.15]. Pass 1 alone would miss the lower bound by 0.04.
- Without qa_1 (descriptive, 240 pairs; RULER repeats the same documents in every qa_1 prompt at this length): −2.19 [−3.33, −1.33]. The gate outcome is the same.
- Server output against the sparse-tail reference (informative, first 64 tokens): most configs match all 64; at 256k `pfsd27648` diverges after 35 tokens and `fork_dense_ub2048` after 27 (different kernels or ubatching change greedy ties; not scored).

## Amendment 4 (read after the data)

Box B's last unit failed one count: prompt id 12004 (length 256,002) has a prefill of 256,001 tokens, which ends in a one-token ubatch. The engine never counts a one-token ubatch (`build_sparse_prefill` requires `nt > 1`); that token runs dense attention, as the d512 tail requires. `pfe_check.expected_counts` counted it, so it expected 16 × 512 dense rows instead of 16 × 511. It is the only such prompt among the 260. The checker now skips one-token ubatches; `e2e_decide.py --amend4` reads that one unit from `tmp/` (where the box left it) and applies every registered unit rule to it. The unit re-checks `TASK OK`. No GPU rerun: the run is deterministic. An independent review (Codex) confirmed the cause and reproduced every quality field.

## Predictions

| | prediction | outcome |
|---|---|---|
| P1 | gates P, K, B, counts, A pass; no row INVALID | **partly held**: all gates passed (P with amendment 1's RAM floor); one unit INVALID under the registered checker, valid after amendment 4 |
| P2 | MTP dense and `-ub 2048` fit at 262,144 on the 5090 | held (27.7 and 26.1 GiB); peaks against the memory table not compared |
| P3 | 256k `fork_dense` TTFT 190–250 (218); `pfsd27648` 80–110 (90); ratio 1.9–2.8 | held: 218.4, 92.3, 2.37 |
| P4 | 256k decode: dense 30–36, `pfsd27648` 55–68 (ratio 1.7–2.0), MTP 45–70 tok/s | dense 35.4 held; `pfsd27648` 68.2, just above the range; ratio 1.93 held; MTP 50.2 (fork), 53.4 (upstream) held |
| P5 | R_256 (5090) 1.7–2.6, point 2.2 | **missed: 1.61** (the best dense was upstream, ~1.43× faster than the fork base at 256k; against `fork_dense` 2.31) |
| P6 | R_128 (5090) 1.35–1.80, point 1.55 | **missed: 1.27** (same cause; best dense `up_mtp`) |
| P7 | R_256 − R_128 ≥ 0.3 | held: +0.34 |
| P8 | best dense at 256k non-MTP (p 0.6); `fork_dense_ub2048` best in ≥ 1 block at each length (p 0.6) | first held (`up_dense`); second missed (upstream won every block) |
| P9 | R_256(55296) 0.03–0.12 below 27648; `pf_only` 0.15–0.45 below | first missed (0.023 below); second held (0.19) |
| P10 | R_128 (4090) 1.10–1.35 (1.22); TTFT ratio 1.25–1.40, cleared p 0.55 | held: 1.33; 1.40, cleared |
| P11 | dense mean 70–92; primary Δ −5 to +1, PASS_27648 p 0.55; 55296 Δ ≥ 27648 Δ − 0.5; qa_1 dense at 256k below 85 | dense 95.3 **missed** (above); Δ −2.40 held, PASS_27648 held (with amendment 4); 55296 −1.88 held; qa_1 85.0 **missed** (equal, not below) |
| P12 | gate N passes; MTP acceptance 0.55–0.85 | gate N held; acceptance 0.52–0.58 on the 5090 (mostly below), 0.557 on the 4090: partly missed |

## Deviations

1. Amendment 1 (before data): RAM floor 28 GiB for the 5090 boxes (30 GiB offers); no memory trouble seen.
2. Amendments 2 and 3 (before arm D data): box D's network failed twice in setup (model download, GitHub clone); arm D relaunched on the same box from on-box git mirrors of the first run's verified clones, timeout 8400 s.
3. Amendment 4 (after the data): the count rule for one-token ubatches; the quality outcome depends on it.
4. The 4090 ran at a 330 W power limit (stock 450 W); its absolute times are slower than a stock card's.

## What it settles, and what it does not

Settled (one 5090, one 4090, one model file, one book prompt at two lengths): at ~260k on the 5090, the combined path is 1.61× faster end to end than the best dense measured in llama.cpp (upstream master, `-ub 512`), and passes the RULER gate at 256k with q8_0 KV in the chat format (with amendment 4); the gain grows from 128k (1.27×) to 256k; on a 330 W 4090 at 128k it is 1.33×.

Not settled: other cards and power limits, weights, models, thinking-on use, f16 KV at 256k, generation lengths other than 1,000, batched serving, the fork rebased on upstream (the sparse path ran on the older base here; see `cuda_5090_pfrebase_dev/` for a development run after the rebase), and RULER quality on the 4090.

Consequence (registered, MODERATE band): reported as measured; no headline beyond the number.

## Files

- `decision.json`, `decision_as_registered.json`: the decisions (the `run`/`runs` fields are the project repository's relative paths; `scripts/pf_e2e/decide.sh` rebuilds that layout).
- `manifest.json`: the RULER manifest as run (paths relative to the project repository's `results/pf_e2e/analysis/`; here `vastai49` and `vastai50` are under `cuda_5090_pfe2e_ruler/`).
- `inputs/`: metadata of the two prompt files (`pfe_book_q38.i32.meta.json`, `pfe_ruler256_q38.i32.meta.json` with the RULER references used for scoring, `.prov.json`), their SHA-256s (`sha256.txt`), the RULER commit and input hashes.
- Not committed: the token files `pfe_book_q38.i32` (1.5 MB, PG19 books 1–2 in Qwen3.8 tokens) and `pfe_ruler256_q38.i32` (270 MB).
