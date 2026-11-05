# When does sparse decode pay on consumer GPUs?

**Status: draft, not public. RULER and wall-clock results pending.**

Training-free page-sparse decoding for ordinary dense LLMs inside llama.cpp, measured on RTX 4090 and RTX 3090. The long-context speedup was predicted from short-context measurements before it was measured.

## Results in one table

Single-sequence decode speed at a 128k-token context. Prefill is dense in both cases.

| GPU | Model (KV cache) | Dense tok/s | Sparse tok/s | Speedup | Predicted beforehand |
|---|---|---|---|---|---|
| RTX 4090 | Llama-3.2-3B (f16) | 48.7 | 75.4 | **1.55×** | 1.6 (band 1.45–1.75) |
| RTX 4090 | Llama-3.1-8B (q8_0) | 39.7 | 52.1 | **1.31×** | 1.30 (band 1.15–1.45) |
| RTX 3090 | Llama-3.2-3B (f16) | 44.7 | 60.4 | **1.35×** | 1.27 (calibrated ≤ 32k) |
| RTX 3090 | Llama-3.1-8B (q8_0) | 21.9 | 38.8 | **1.77×** | 1.70 (calibrated ≤ 32k) |
| RTX 3090 | Qwen3-8B (q8_0) | 19.9 | 36.0 | **1.81×** | 1.75 (calibrated ≤ 32k) |

- The sparse path reads about 21–22% of the cached keys at 128k.
- Below about 32k context, sparse decoding is slower than dense, so it switches on only above a threshold (default 32768).

## What is sparse here

At each decode step, every KV head attends to:
- the first 64 cells;
- the most recent cells;
- the highest-scoring 64-cell pages, up to a budget B (27648 cells at 128k above).

Pages are scored from per-page key statistics (mean and variance) kept in the KV cache. This is the Quest/MOIRA family of page selection; nothing is trained. The contribution here is a working implementation for dense models in llama.cpp (CUDA), measured end to end, and a cost model that predicts when it pays.

## Quality at the operating point

The budget was chosen for a mean next-token KL divergence of about 0.02 against dense decoding (PG19 books, 256 teacher-forced tokens per book).

| Check | Llama-3.2-3B | Llama-3.1-8B | Qwen3-8B |
|---|---|---|---|
| In-engine KL vs the Python reference, 32k | ratio 1.02 | — | — |
| In-engine KL vs the Python reference, 128k | ratio 1.05 | ratio 1.30 | no reference |
| KL at 32768 cells, 128k | 0.018 | 0.017 | 0.025 |
| Top-1 token changed vs dense, 32768 cells | 4.5% | 6.2% | 10.0% |
| KL / top-1 change at 27648 cells (the 8B and Qwen3 speed budget) | — | — | 0.031 / 11.3% |
| Needle retrieval at the speed budget, 128k (single / multikey) | 100% / 94% | 100% / 100% | 89% / 100% |
| RULER, 13 tasks, 128k | — | *pending* | — |

The speed budgets (27648 cells for the 8B models, 28672 for the 3B) are slightly below 32768. For Llama the KL there interpolates to about 0.02; for Qwen3 it is 0.031, measured.

KL vs dense does not mean "no quality loss". Read the top-1 change rate and the RULER result before relying on it.

## How the speedup is predicted

Per decoded token:

```
t_sparse(T) = t_dense(T) − L · [a_dense(T) − a_sparse(T, B)] + o0 + o1·T
```

- `a` is the per-layer attention kernel time (microbenchmark).
- `L` is the number of layers.
- `o0 + o1·T` is the cost of selecting pages, fitted on the target GPU at 8k–32k.

On the RTX 3090 the predictions were committed before the 64k and 128k runs. All 6 landed within ±10% of measurement; the worst miss was 6.2%.

## What did not work, and what failed its gate

Every test was pre-registered (see [protocols/](protocols/)). Several failed:

- **The first CUDA build gave only 1.05–1.18× at 128k.** Selection overhead was the cause: two ggml kernels (`pool_1d` with a whole-row window, and `top_k_mask`) used too few GPU threads. Fixing them gave the numbers above.
- **A kernel microbenchmark (C2) was mis-designed.** It picked pages independently per head, while the engine shares pages across a GQA group. Its KILL verdict is recorded as is.
- **Qwen3-8B at the Llama budget narrowly failed its quality gates:** KL 0.031 against a 0.03 limit, and single-key retrieval at 89% against 90%. Budgets should be set per model family.
- **The 8B in-engine KL was 30% above the Python reference** (gate ≤ 25%). The likely causes are q8_0 weights and TF32 in cuBLAS scoring; this is not resolved.

## Try it

*Fork branch link and build instructions to be added.*

```
LLAMA_SPARSE_DECODE=27648 ./llama-cli -m model.gguf -c 131072 -fa on ...
```

Options:
- `LLAMA_SPARSE_DECODE_MIN_CTX` (default 32768): the cache length below which decoding stays dense.
- `LLAMA_SPARSE_DECODE_SCORE=mv|quest`: the page score.

## Repository

- `protocols/`: the pre-registrations, with the commit and time each was committed.
- `results/`: raw results (JSONL) and analyses.
- `scripts/`: run and analysis scripts.

## Scope and limits

- Batch size 1, decode only. Prefill is unchanged and dominates wall-clock at 128k (see the wall-clock result, *pending*).
- Two GPU models (RTX 4090 48 GB variant, RTX 3090), two model families (Llama 3.x, Qwen3).
- Quality: KL against dense on 4 PG19 books, 40 synthetic needle trials per model, RULER (*pending*).
- Metal (Apple silicon): kernel-level results only so far.
