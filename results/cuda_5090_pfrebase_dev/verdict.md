# Development run, not a pre-registered test: the sparse path rebased onto upstream c6a25fc62 (RTX 5090, Qwen3.8-27B UD-Q4_K_M, KV q8_0)

No protocol governs this run; it was made after PF-E2E (`results/pfe2e_analysis/verdict.md`) to check whether upstream's faster dense path takes away the gain. Nothing here was predicted beforehand, and nothing here changes a PF-E2E outcome. On-box files of box vastai52, run 20261122T221857Z.

**Code.** The sparse path (the 39 commits of the PF-E2E fork build) rebased onto ggml-org/llama.cpp c6a25fc62, the upstream commit PF-E2E measured, tree **60da1b333ec742f37fd674dec04be0f64dbfb44e**, checked on the box (`SOURCE OK`). The same tree is branch [`pf-lists-rebase-20261122`](https://github.com/topppg/llama.cpp/tree/pf-lists-rebase-20261122) of the fork (head 75626732c). Upstream c6a25fc62 built next to it.

**Box.** RTX 5090 32,607 MiB, 500 W (default 600 W), driver 580.178.04, CUDA 13.0, 30 GiB RAM, 16 cores. Every setup gate passed (gate K all eight filters), gate B PASS at 256k and 128k (GATHER=2 logits byte-identical to GATHER=1 on this build, `bit/bin_sha256.txt`), references with the same selective-conversion counts as PF-E2E (31,936 and 15,680 per prefill), final line COMPLETE.

**Design.** PF-E2E arm A's script with three changes: the source (c6a25fc62 + the rebased patches, tree checked), 11 fixed rows, and **one block** with 2 timed requests per row (PF-E2E: two blocks, the lower ratio quoted). Same book prompt, model, server flags and environment.

| length | config | wall s | TTFT s | decode tok/s | ratio to the fastest dense here | PF-E2E block 1 wall |
|---|---|---|---|---|---|---|
| 256k | `up_dense` (fastest dense) | 177.28 | 149.66 | 36.28 | 1.000 | 172.45 |
| 256k | `fork_dense` (rebased, sparse off) | 178.33 | 149.64 | 34.93 | 0.994 | 246.72 |
| 256k | `pfsd27648` | 107.27 | 92.40 | 67.55 | **1.653** | 107.01 |
| 256k | `pfsd55296` | 108.85 | 92.29 | 60.61 | 1.629 | 108.55 |
| 256k | `pf_only` | 121.73 | 92.70 | 34.53 | 1.456 | 121.09 |
| 128k | `up_mtp` (fastest dense) | 68.72 | 55.66 | 76.75 | 1.000 | 67.31 |
| 128k | `up_dense` | 72.05 | 52.37 | 50.90 | 0.954 | 70.47 |
| 128k | `fork_dense` | 72.74 | 52.39 | 49.22 | 0.945 | 87.23 |
| 128k | `pfsd27648` | 53.49 | 38.93 | 68.84 | **1.285** | 52.95 |
| 128k | `pfsd55296` | 55.22 | 38.96 | 61.64 | 1.244 | 54.58 |
| 128k | `pf_only` | 59.81 | 39.12 | 48.41 | 1.149 | 59.06 |

Medians of the two timed requests (`e2e/*.jsonl`; the same values are in the `E2E` lines of `status.txt`).

**Reading (post hoc, development run).**
- The rebase gives the fork upstream's dense prefill: `fork_dense` TTFT 149.6 s, equal to `up_dense`, against 218.4 s on the old base.
- The sparse rows hardly moved: at 256k they are within 0.6% of PF-E2E's (+0.24% to +0.53%); at 128k they are 1.0% to 1.3% slower (`pfsd27648` 53.49 against 52.95 s, `pfsd55296` 55.22 against 54.58, `pf_only` 59.81 against 59.06). The rebase did not speed up the sparse path.
- Against upstream dense the combined setup is 1.65× at 256k and 1.28× at 128k. Dense on this box is 2–3% slower than in PF-E2E (`up_dense` 256k +2.8%, `up_mtp` 128k +2.1%), which lifts these ratios slightly; with one block and one box this is not a registered number.
- Our reading, not tested: the time upstream saved in dense prefill is in its dense attention kernel, which the sparse path does not use.
- Also seen, not investigated: the rebased fork's dense decode is ~4% slower than upstream's with every sparse variable unset (34.9 against 36.3 tok/s at 256k, 49.2 against 50.9 at 128k).

Files: `status.txt`, `e2e/` (client rows, server logs, client log), `ref/*.jsonl`, `bit/` (gate B rows and logit SHA-256s), `commit.txt` (the rebased commits on c6a25fc62), `commit_upstream.txt`, `sha.txt`, `matrix.txt`, `stages.txt`, `gpu.txt`. Not committed: build and cmake logs, `tbo_*.txt` gate K outputs (PASS lines in `status.txt`), `gpu_log.csv`, the llama.cpp logs `ref/*.err` and `bit/*.err`, `pgids`, the empty `dl_q38.log`. The development script itself is in the project repository, not here.
