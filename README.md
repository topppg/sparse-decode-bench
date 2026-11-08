# When does sparse decode pay on consumer GPUs?

Giorgio Sarno, Controra Research

**Status: draft, not public.**

Training-free page-sparse decoding for ordinary dense LLMs inside llama.cpp, measured on RTX 4090 and RTX 3090. The long-context speedup was predicted from short-context measurements before it was measured.

**In short:**
- At 128k context, decode is 1.3–1.8× faster on Llama-3.1-8B and Qwen3-8B.
- With the quest page score, RULER at 128k (13 tasks) holds at that speed: Llama-3.1-8B scores **63.6 against 64.7 dense** (Δ −1.1, 95% CI [−4.3, +2.0]) and Qwen3-8B **60.2 against 59.7** (Δ +0.4 [−2.3, +3.3]). The first page score (mv) lost 7.4 points on Llama-3.1-8B at the same budget and speed.
- Llama-3.2-3B decodes 1.35–1.55× faster at its budget but **fails** RULER there (Δ −4.5 [−7.1, −1.9]), so its speed rows are not quality-safe as given.
- The gain grows with context: at 256k, UltraLong-8B decodes **1.86× faster** (23.0 → 42.8 tok/s), and on six multi-needle RULER tasks the loss at that budget is no larger than at 128k (−4.0 against −6.9 points).
- Prefill is unchanged, so a single 1k-token answer over a 128k prompt finishes only 5–15% sooner. The gain is in long generations.
- It is a dial between speed and quality, with a cost model that tells you where you are.

## Results in one table

Single-sequence decode speed at a 128k-token context. Prefill is dense in both cases.

| GPU | Model (KV cache) | Dense tok/s | Sparse tok/s | Speedup | Predicted beforehand | RULER Δ at this budget (quest) |
|---|---|---|---|---|---|---|
| RTX 4090 | Llama-3.2-3B (f16) | 48.7 | 75.4 | 1.55× | 1.6 (band 1.45–1.75) | −4.5, **fails** |
| RTX 4090 | Llama-3.1-8B (q8_0) | 39.7 | 52.1 | **1.31×** | 1.30 (band 1.15–1.45) | −1.1, passes |
| RTX 3090 | Llama-3.2-3B (f16) | 44.7 | 60.4 | 1.35× | 1.27 (calibrated ≤ 32k) | −4.5, **fails** |
| RTX 3090 | Llama-3.1-8B (q8_0) | 21.9 | 38.8 | **1.77×** | 1.70 (calibrated ≤ 32k) | −1.1, passes |
| RTX 3090 | Qwen3-8B (q8_0) | 19.9 | 36.0 | **1.81×** | 1.75 (calibrated ≤ 32k) | +0.4, passes |

- Budgets: 27648 cells for the 8B models, 28672 for the 3B. The sparse path reads about 21–22% of the cached keys at 128k.
- RULER was measured once per model, on an RTX 4090, with the gate Δ ≥ −3 points and 95% lower bound ≥ −6 (see Quality below). The 3B fails it at this budget; a larger 3B budget has not been tested.
- These runs used the mv page score. quest runs at the same speed: 51.94 against 52.05 tok/s for mv (Llama-3.1-8B, RTX 4090 24 GB, 128k).
- Below about 32k context, sparse decoding is slower than dense, so it switches on only above a threshold (default 32768).

## What is sparse here

At each decode step, every KV head attends to:
- the first 64 cells;
- the most recent cells;
- the highest-scoring 64-cell pages, up to a budget B (27648 cells at 128k above).

Pages are scored from per-page key statistics kept in the KV cache. The default score is **quest**: the query's positive part times each page's per-dimension key maximum, plus its negative part times the minimum, an upper bound on any key's score in the page. The alternative **mv** scores a page by its key mean and variance. Nothing is trained.

This is the page selection of Quest [1] (the quest score) and MOIRA [2] (page statistics; mv is a moment estimate in that spirit). The contribution here is a working implementation for dense models in llama.cpp (CUDA), measured end to end, and a cost model that predicts when it pays.

## Quality at the operating point

### RULER, 128k, Llama-3.1-8B q8_0, 27648 cells (13 tasks × 20 samples)

| task | dense | quest | mv |
|---|---|---|---|
| niah_single_1 / 2 / 3 | 100 / 100 / 90 | 100 / 100 / 90 | 100 / 100 / 95 |
| niah_multikey_1 / 2 / 3 | 85 / 25 / 30 | 85 / 15 / 20 | 75 / 15 / 20 |
| niah_multivalue | 72.5 | 77.5 | 42.5 |
| niah_multiquery | 86.2 | 86.2 | 65.0 |
| variable tracking | 86 | 82 | 74 |
| cwe / fwe | 1 / 80 | 0.5 / 80 | 1 / 76.7 |
| qa_1 / qa_2 | 55 / 30 | 60 / 30 | 55 / 25 |
| **mean** | **64.7** | **63.6** | **57.2** |
| Δ vs dense [95% CI] | — | **−1.1 [−4.3, +2.0]** | −7.4 [−11.0, −4.1] |

- The pre-registered gate (Δ ≥ −3, lower bound ≥ −6) **passes with quest** and fails with mv.
- Dense outputs are token-identical across the three RULER runs (260/260).
- The remaining losses are on multi-needle retrieval: −10 on multikey_2 and multikey_3 (2 of 20 prompts each), −4 on variable tracking.

### RULER on Qwen3-8B and Llama-3.2-3B (PORT-RQ)

The same 13 tasks × 20 samples at 128k, the same gate, quest at the speed-table budgets, measured on the released build (e7059e52a). Pre-registered before any data, with the consequence for this README fixed in advance.

| model (KV cache) | budget | dense | quest | Δ [95% CI] | gate |
|---|---|---|---|---|---|
| Qwen3-8B-Base, YaRN to 128k (q8_0) | 27648 | 59.7 | 60.2 | +0.4 [−2.3, +3.3] | **pass** |
| Llama-3.2-3B (f16) | 28672 | 68.2 | 63.7 | −4.5 [−7.1, −1.9] | **fail** |

- Both predictions were PASS: the 3B prediction was wrong, and Qwen's Δ landed just above its predicted interval [−4, 0].
- The 3B loses on multi-needle tasks: multikey_2 −30 (6 of 20 prompts), multivalue −12.5, multiquery −10, fwe −6.7. Single needles and QA hold.
- Read after the data, not pre-registered: on multikey_2, all 7 lost prompts are near-misses, a number sharing its first 4–6 digits with the answer (5149787 → 5149780). In 3 of 20 prompts the answer crosses a 64-cell page edge, and all 3 are among the losses. The other 4 do not, so page edges explain at most part of it. Not tested yet.
- The 3B had passed its earlier checks with mv (KL 0.018 at 32768 cells; needles 100% / 94% at the speed budget), another case where those checks did not predict RULER.
- Per-task scores: `results/cuda_4090_portrq/`.

### Why mv failed (PORT-D)

On the 120 prompts of the six tasks that lost points with mv, each candidate cause was tested separately:

| change | effect on the six-task mean [95% CI] |
|---|---|
| **page score quest instead of mv** | **+12.4 [+6.3, +18.8]** |
| first 2 layers dense | −2.8 [−6.0, 0.0] |
| f32 page statistics instead of f16 | −1.7 [−4.3, +0.2] |

Only the page score matters. A likely reading: a page's mean dilutes a single relevant key among 64, while quest's per-dimension bound keeps it. This was not tested directly.

### Larger budgets with mv (reported for the dial)

RULER at 128k, Llama-3.1-8B, mv: −1.9 [−4.5, +0.6] at 49152 cells and −1.3 [−3.4, +0.7] at 65536. Both pass, at a lower speed (1.23× and 1.11× on the RTX 4090 24 GB).

### KL against dense and needle retrieval (mv score)

The budget was first chosen for a mean next-token KL divergence of about 0.02 against dense decoding (PG19 books, 256 teacher-forced tokens per book). These checks used the mv score.

| Check | Llama-3.2-3B | Llama-3.1-8B | Qwen3-8B |
|---|---|---|---|
| In-engine KL vs the Python reference, 32k | ratio 1.02 | — | — |
| In-engine KL vs the Python reference, 128k | ratio 1.05 | ratio 1.30 | no reference |
| KL at 32768 cells, 128k | 0.018 | 0.017 | 0.025 |
| Top-1 token changed vs dense, 32768 cells | 4.5% | 6.2% | 10.0% |
| KL / top-1 change at 27648 cells (the 8B and Qwen3 speed budget) | — | — | 0.031 / 11.3% |
| Needle retrieval at the speed budget, 128k (single / multikey) | 100% / 94% | 100% / 100% | 89% / 100% |

KL ≈ 0.02 coexisted with a 7.4-point RULER loss under mv, so **KL against dense is not a sufficient quality measure**. Check multi-needle tasks.

## Longer context: 256k

UltraLong-8B (`nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`, q8_0, KV q8_0), RTX 4090 48 GB, quest.

| context | budget | dense tok/s | sparse tok/s | speedup |
|---|---|---|---|---|
| 128k | 49152 | 40.5 | 47.1 | 1.16× |
| 256k | 49152 | 23.0 | 42.8 | **1.86×** |
| 256k | 98304 | 23.0 | 33.4 | 1.45× |

The same budget gives a larger speedup at a longer context, because dense decode slows with the cache while the sparse read stays fixed.

**Does quality hold at a fixed budget?** Pre-registered test: on the six multi-needle RULER tasks, Δ(256k) − Δ(128k) ≥ −3 points at 49152 cells.
| context | dense | quest:49152 | Δ [95% CI] | quest:98304 | Δ [95% CI] |
|---|---|---|---|---|---|
| 128k | 82.4 | 75.5 | −6.9 [−12.0, −2.2] | — | — |
| 256k | 67.4 | 63.4 | −4.0 [−8.7, +0.3] | 63.4 | −4.0 [−7.9, −0.2] |

- **The test passes:** Δ(256k) − Δ(128k) = +2.9, 95% CI [−3.7, +9.7]. The wide interval rules out a large degradation, not a small one.
- Dense itself loses 15 points from 128k to 256k, so there is less to lose at 256k. This does not show that sparse decoding improves with context.
- Doubling the budget to 98304 cells (the same fraction of the cache as 49152 at 128k) does not help. The loss sits in a few multi-key prompts.
- The six-task mean is not comparable to the 13-task numbers above: it contains only the hard tasks.

KL against dense at a fixed budget rises with context (49152 cells: 0.017 at 128k, 0.023 at 256k; 98304 cells: 0.006 and 0.013).

## The speed–quality dial

Decode speedup at 128k on the RTX 3090 by budget (cells per KV head per step):

| budget | 8192 | 16384 | 27–28k | 32768 | 49152 |
|---|---|---|---|---|---|
| Llama-3.2-3B | 1.75× | 1.56× | 1.34× | 1.27× | 1.07× |
| Llama-3.1-8B | 2.25× | 2.01× | 1.76× | 1.67× | 1.42× |
| Qwen3-8B | 2.30× | 2.07× | 1.80× | 1.70× | 1.43× |

For the 8B, KL against dense (mv) is about 0.11, 0.05, 0.02 and 0.017 at 8192, 16384, 27648 and 32768 cells. RULER was measured at 27648 (quest and mv), 49152 and 65536 (mv).

## Wall-clock, prefill included

128k-token prompt plus 1024 generated tokens, RTX 3090:

| model | dense | sparse | end-to-end |
|---|---|---|---|
| Llama-3.2-3B | 98 s | 94 s | 1.05× |
| Llama-3.1-8B | 147 s | 129 s | 1.14× |
| Qwen3-8B | 163 s | 142 s | 1.15× |

The gain grows with output length. It matters most when one long context is reused for long generations (agents, long reports), not for single short answers.

## How the speedup is predicted

Per decoded token:

```
t_sparse(T) = t_dense(T) − L · [a_dense(T) − a_sparse(T, B)] + o0 + o1·T
```

- `a` is the per-layer attention kernel time (microbenchmark).
- `L` is the number of layers.
- `o0 + o1·T` is the cost of selecting pages, fitted on the target GPU at 8k–32k.

On the RTX 3090 the predictions were committed before the 64k and 128k runs. All 6 landed within ±10% of measurement; the worst miss was 6.2%.

At 256k (8× beyond calibration, RTX 4090 48 GB) the model failed its ±10% gate: 5 of 8 points within 10%. It over-predicted at 128k by 5–7% and under-predicted at 256k by 7–15%, so it is conservative where the gain is largest.

## What did not work, and what failed its gate

Every test was pre-registered (see [protocols/](protocols/)). Several failed:

- **The first page score (mv) lost 7.4 RULER points** at the KL-chosen budget, concentrated in multi-needle tasks. Dense early layers and f32 statistics did not help; the quest score fixed it at the same speed.
- **The first CUDA build gave only 1.05–1.18× at 128k.** Selection overhead was the cause: two ggml kernels (`pool_1d` with a whole-row window, and `top_k_mask`) used too few GPU threads. Fixing them gave the numbers above.
- **The `top_k_mask` fix first made prefill-sized batches up to 2.2× slower.** The final version keeps master's kernel when rows fill the GPU: decode cases at a 128k cache are 1.9–2.0× faster and no case is more than 0.6% slower.
- **A kernel microbenchmark (C2) was mis-designed.** It picked pages independently per head, while the engine shares pages across a GQA group. Its KILL verdict is recorded as is.
- **Qwen3-8B at the Llama budget narrowly failed its quality gates:** KL 0.031 against a 0.03 limit, and single-key retrieval at 89% against 90%. Budgets should be set per model family. (With quest it later passed RULER at that budget, +0.4.)
- **Llama-3.2-3B fails RULER at its speed budget** (quest, 28672 cells): −4.5 [−7.1, −1.9], although its KL and needle checks had passed. No larger budget was tested before release.
- **The 8B in-engine KL was 30% above the Python reference** (gate ≤ 25%). The likely causes are q8_0 weights and TF32 in cuBLAS scoring; this is not resolved.
- **The cost model missed its 256k gate** (above).

## Try it

CUDA only. Build the [`sparse-decode` branch](https://github.com/topppg/llama.cpp/tree/sparse-decode) of the fork (e7059e52a, on llama.cpp master 6023692b1):

```
git clone -b sparse-decode https://github.com/topppg/llama.cpp llama.cpp-sparse && cd llama.cpp-sparse
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc) --target llama-completion llama-bench
```

Generate with a budget of 27648 cells (all layers and the KV cache on the GPU):

```
LLAMA_SPARSE_DECODE=27648 ./build/bin/llama-completion -m model.gguf -c 131072 -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -f long_prompt.txt -n 1024 -no-cnv
```

Measure decode speed at a 128k cache as in the table above (run once without the variable for dense):

```
LLAMA_SPARSE_DECODE=27648 ./build/bin/llama-bench -m model.gguf -fa 1 -p 0 -n 64 -r 3 -d 131072 -ctk q8_0 -ctv q8_0
```

Options:
- `LLAMA_SPARSE_DECODE=<cells>` (at least 144; 0 or unset is dense): the budget per KV head per decode step. 27648 was used for the 8B models at 128k and 49152 at 256k.
- `LLAMA_SPARSE_DECODE_MIN_CTX` (default 32768): the cache length below which decoding stays dense.
- `LLAMA_SPARSE_DECODE_SCORE=quest|mv` (default quest): the page score.

Limits:
- With the variable set, the context refuses to start unless all layers and the KV cache are on CUDA devices.
- e7059e52a is 5cf439778 (the build checked against the research build, `results/cuda_4090_48gb_featcheck`) plus that check. These commands were run as written on an RTX 3090 with Llama-3.2-3B-Instruct Q8_0: 34.7 -> 59.2 tok/s at a 128k cache, KV q8_0.
- One sequence, decode only; prefill is dense.
- On Apple silicon (Metal) only kernel-level results exist so far: a decode speedup of 1.56x at 128k, projected from kernel microbenchmarks on an M4 Pro.

## Repository

- `protocols/`: the pre-registrations, with the commit and time each was committed.
- `results/`: raw results (JSONL) and analyses.

## Scope and limits

- Batch size 1, decode only. Prefill is unchanged and dominates wall-clock at 128k (see Wall-clock above).
- Three GPUs (RTX 3090, RTX 4090 24 GB, RTX 4090 48 GB), two model families (Llama 3.x, Qwen3).
- Quality:
  - RULER, 13 tasks × 20 samples at 128k, on Llama-3.1-8B (quest and mv), Qwen3-8B and Llama-3.2-3B (quest);
  - the six multi-needle RULER tasks at 128k and 256k on UltraLong-8B;
  - KL against dense on 4 PG19 books;
  - 40 synthetic needle trials per model.
- Multi-needle retrieval is the weak spot. Finding a single fact loses nothing on all three models, and the non-needle tasks hold. Retrieving several similar keys or values loses 0–3 points on the 8B models and 11.5 on the 3B (mean of the five multi-needle RULER tasks). On Llama-3.1-8B part of this is hidden: dense already scores 25–30 on multikey_2 and multikey_3. This is a known weakness of page-level selection on synthetic retrieval [3, 4].
- RULER is synthetic. Real long-document tasks are covered only by its two QA tasks.
- Sparse decode output is not bitwise reproducible from run to run (dense is): on 30 RULER prompts, two quest runs give token-identical outputs on 21–24 of 30, with the same scores. The released branch matches the research build within that spread.

## References

1. Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)
2. Dich Nhat Minh Nguyen, Tran Dang Duong Nguyen. *MOIRA: Mass-Oriented Indexing with Ragged Attention for Long-Context Decoding.* 2026. [arXiv:2610.04313](https://arxiv.org/abs/2610.04313)

3. Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, Edoardo M. Ponti. *The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs.* Findings of ACL 2026. [arXiv:2504.17768](https://arxiv.org/abs/2504.17768) (§4.2: Quest degrades on synthetic RULER NIAH; page granularity amplifies it.)
4. Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, Lili Qiu. *RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval.* 2024. [arXiv:2409.10516](https://arxiv.org/abs/2409.10516) (Table 9: Quest at 2K tokens, 128k RULER, keeps single needles and loses on multi-key, multi-query and multi-value.)

## Licence

Text, protocols and results: [CC BY 4.0](LICENSE). Scripts: [MIT](LICENSE-MIT). The llama.cpp fork is MIT, as llama.cpp.
