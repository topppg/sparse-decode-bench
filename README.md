# When does sparse decode pay on consumer GPUs?

Training-free page-sparse decoding for ordinary GQA LLMs (dense and MoE) inside llama.cpp (CUDA), measured on RTX 3090, RTX 4090 and RTX 5090. The first long-context speedups were predicted before they were measured (for the mv page score): dense speed and attention kernel times were measured at the target length, and only the cost of page selection was extrapolated from 8k–32k.

**In short:**
- At 128k context with the quest page score (the default), decode is **1.72× faster on Qwen3-30B-A3B** at Q4_K_M on an RTX 5090 (MoE; 1.41× at Q8_0 on an RTX 4090 48 GB) and 1.64–1.68× faster on Llama-3.1-8B and Qwen3-8B (RTX 3090). The speedup depends on the card: with the earlier mv score, which runs at the same speed, the 8Bs got 1.77–1.81× on another RTX 3090 and Llama-3.1-8B 1.31× on an RTX 4090 48 GB, so on the same kind of card the MoE gains more than the dense 8B.
- RULER at 128k (13 tasks) holds with quest on three models: Qwen3-30B-A3B scores **86.8 against 87.8 dense** at Q4_K_M (Δ −1.0, 95% CI [−2.2, −0.02]) and **86.6 against 87.2** at Q8_0 (Δ −0.6 [−1.9, +0.4]), Llama-3.1-8B **63.6 against 64.7** (Δ −1.1 [−4.3, +2.0]) and Qwen3-8B **60.2 against 59.7** (Δ +0.4 [−2.3, +3.3]).
- Llama-3.2-3B decodes 1.35–1.55× faster but **fails** RULER at that budget (Δ −4.5 [−7.1, −1.9]), so its speed rows are not quality-safe as given.
- Most of the loss is in multi-needle retrieval (on the Q4_K_M MoE, the small loss is in QA instead). Finding a single fact loses nothing on any model.
- The gain grows with context: at 256k, UltraLong-8B decodes **1.86× faster** (23.0 → 42.8 tok/s).
- Prefill is unchanged, so a single 1k-token answer over a 128k prompt takes only 4–13% less time. The gain is in long generations.
- The full KV cache stays in VRAM. This makes decode faster; it does not let a longer context fit.

## How it works

The KV cache is split into **pages** of 64 consecutive **cells** (one cell is one cached token position). At each decode step, every KV head attends to:
- the first page;
- the most recent cells (at most 128, aligned to pages);
- the highest-scoring pages.

The **budget** B covers all three: about (B − 80)/64 pages are picked, so about B cells are read per KV head per step.

Pages are scored from per-page key statistics kept next to the KV cache. All query heads of a GQA group share one set of pages. Nothing is trained, and no cell is evicted: the full cache stays in memory, plus the statistics (about 3% of the KV cache at f16, 6% at q8_0).

Two page scores are implemented:
- **quest** (the default): the query's positive part times each page's per-dimension key maximum, plus its negative part times the minimum. This is an upper bound on any key's score in the page.
- **mv**: a page's key mean and variance. This was the first score; it lost quality on RULER (see below).

Sparse decoding is slower than dense below about 32k context, so it switches on only above a threshold (default 32768 cells).

This is the page selection of Quest [1] (the quest score) and MOIRA [2] (page statistics; mv is a moment estimate in that spirit). The contribution here is a working implementation for dense and MoE GQA models in llama.cpp, measured end to end, and a cost model that predicts when it pays.

## Speed

Single-sequence decode speed at a 128k-token context with the quest page score, on the release build (e7059e52a). Weights are Q8_0 except the RTX 5090 row (Q4_K_M); the KV cache type is in brackets. `llama-bench`, 64 generated tokens, 5 repetitions.

| GPU | Model (KV cache) | Dense tok/s | Sparse tok/s | Speedup | RULER Δ at this budget (quest) |
|---|---|---|---|---|---|
| RTX 5090 | Qwen3-30B-A3B-Instruct-2507, MoE, Q4_K_M (q8_0) | 46.1 | 79.2 | **1.72×** | −1.0, passes |
| RTX 4090 48 GB | Qwen3-30B-A3B-Instruct-2507, MoE (q8_0) | 37.4 | 52.6 | **1.41×** | −0.6, passes |
| RTX 3090 | Qwen3-8B-Base, YaRN (q8_0) | 21.5 | 36.1 | **1.68×** | +0.4, passes |
| RTX 3090 | Llama-3.1-8B (q8_0) | 23.7 | 38.8 | **1.64×** | −1.1, passes |
| RTX 3090 | Llama-3.2-3B (f16) | 44.4 | 61.3 | 1.38× | −4.5, **fails** |

- Budgets: 27648 cells for the 8B models and the MoE, 28672 for the 3B. That is about 21–22% of the cached keys at 128k.
- On the same RTX 3090, quest and mv run at the same speed: quest/mv is 0.99–1.00 at every context from 8k to 128k, on all three dense models (`results/cuda_3090_portgq/`).
- RULER was measured once per model, on an RTX 4090, and for the Q4_K_M MoE on the RTX 5090 (see Quality). No larger 3B budget has been evaluated on RULER.
- The speedup depends on the card. This RTX 3090 (350 W limit) runs the 8B models' dense decode about 8% faster than the one in the table below (300 W), while their sparse speed is the same, so their speedups are lower; the 3B is unchanged. On the RTX 4090 48 GB, Llama-3.1-8B gets 1.31×.
- The MoE speedup was predicted at 1.5–2.3× and came in at 1.41×. A rough breakdown, read after the data and not a measurement: fitting dense time as linear in context (8k, 32k, 128k), reading the KV cache is about 80% of the MoE's dense decode time at 128k, against about 62% for Llama-3.1-8B on an RTX 4090 48 GB (earlier build, mv), so the MoE has more to gain. What the fit does not explain, about 9 ms per token or 0.19 ms per layer, is the sparse step's own overhead (page selection and the sparse attention path); it is about half of the MoE's sparse step. Removing all of it would give at most about 2.7×.
- At Q4_K_M on an RTX 5090 (Blackwell; the fork's first run there) the MoE got 1.72×, inside its pre-registered 1.3–1.9× (`protocols/port-mq4.md`). It is below dense up to 32k (0.66× at 8k, 0.93× at 32k) and 1.24× at 64k. The breakdown above has not been fitted for this run.

### Earlier measurements with the mv score, and their predictions

These rows used the mv page score and were predicted beforehand by the cost model (see How the speedup is predicted).

| GPU | Model (KV cache) | Dense tok/s | Sparse tok/s | Speedup | Predicted beforehand |
|---|---|---|---|---|---|
| RTX 4090 48 GB | Llama-3.2-3B (f16) | 48.7 | 75.4 | 1.55× | 1.6 (band 1.45–1.75) |
| RTX 4090 48 GB | Llama-3.1-8B (q8_0) | 39.7 | 52.1 | 1.31× | 1.30 (band 1.15–1.45) |
| RTX 3090 (300 W) | Llama-3.2-3B (f16) | 44.7 | 60.4 | 1.35× | 1.27 |
| RTX 3090 (300 W) | Llama-3.1-8B (q8_0) | 21.9 | 38.8 | 1.77× | 1.70 |
| RTX 3090 (300 W) | Qwen3-8B-Base, YaRN (q8_0) | 19.9 | 36.0 | 1.81× | 1.75 |

quest was also timed against mv on Llama-3.1-8B on an RTX 4090 24 GB at 128k: 51.94 against 52.05 tok/s.

### Wall-clock, prefill included

128k-token prompt plus 1024 generated tokens, RTX 3090 (300 W), mv score:

| model | dense | sparse | end-to-end |
|---|---|---|---|
| Llama-3.2-3B | 98 s | 94 s | 1.05× |
| Llama-3.1-8B | 147 s | 129 s | 1.14× |
| Qwen3-8B | 163 s | 142 s | 1.15× |

The gain grows with output length. It matters most when one long context is reused for long generations (agents, long reports), not for single short answers.

## Try it

Requirements: an NVIDIA GPU with CUDA, all layers and the KV cache on the GPU, one sequence. With the variable set, the context refuses to start otherwise.

Build the [`sparse-decode` branch](https://github.com/topppg/llama.cpp/tree/sparse-decode) of the fork at the released commit (e7059e52a, on llama.cpp master 6023692b1):

```
git clone -b sparse-decode https://github.com/topppg/llama.cpp llama.cpp-sparse && cd llama.cpp-sparse
git checkout e7059e52a
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc) --target llama-completion llama-bench
```

Generate with a budget of 27648 cells:

```
LLAMA_SPARSE_DECODE=27648 ./build/bin/llama-completion -m model.gguf -c 131072 -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -f long_prompt.txt -n 1024 -no-cnv
```

Measure decode speed at a 128k cache (run once without the variable for dense). To reproduce the speed table, use the table's KV type and budget (f16 and 28672 for the 3B); add `LLAMA_SPARSE_DECODE_SCORE=mv` for the earlier mv rows:

```
LLAMA_SPARSE_DECODE=27648 ./build/bin/llama-bench -m model.gguf -fa 1 -p 0 -n 64 -r 5 -d 131072 -ctk q8_0 -ctv q8_0
```

Options:
- `LLAMA_SPARSE_DECODE=<cells>` (at least 144; 0 or unset is dense): the budget per KV head per decode step. 27648 was used for the 8B models and the MoE at 128k, and 49152 at 256k.
- `LLAMA_SPARSE_DECODE_MIN_CTX` (default 32768): the cache length below which decoding stays dense.
- `LLAMA_SPARSE_DECODE_SCORE=quest|mv` (default quest): the page score.

These commands were run as written on an RTX 3090 with Llama-3.2-3B-Instruct Q8_0: 34.7 → 59.2 tok/s at a 128k cache, KV q8_0. This checks that the build works; quality was not evaluated at that setting. On Apple silicon (Metal) only kernel-level results exist so far: a decode speedup of 1.56× at 128k, projected from kernel microbenchmarks on an M4 Pro.

## Quality

**RULER** is a synthetic long-context benchmark. Its 13 tasks:
- `niah_single_1–3`: find one needle (a key and its value) in filler text;
- `niah_multikey_1–3`: find the value of one key among several similar keys (in 2 and 3 the filler is itself made of decoy keys);
- `niah_multivalue`: return all four values of one key;
- `niah_multiquery`: return the values of four keys;
- `vt`: follow chains of variable assignments;
- `cwe`: list the 10 most common words in a long word list; `fwe`: list the 3 most frequent words in a coded text;
- `qa_1` / `qa_2`: question answering over documents (SQuAD / HotpotQA).

Each task has 20 prompts, scored 0–100. Δ is the sparse score minus the dense score, in points, averaged over the tasks. The 95% CI comes from a paired bootstrap over prompts (10 000 resamples). The gate, fixed before each run: Δ ≥ −3 and the lower bound of the CI ≥ −6.

### Llama-3.1-8B, 128k, 27648 cells

| task | dense | quest | mv |
|---|---|---|---|
| niah_single_1 / 2 / 3 | 100 / 100 / 90 | 100 / 100 / 90 | 100 / 100 / 95 |
| niah_multikey_1 / 2 / 3 | 85 / 25 / 30 | 85 / 15 / 20 | 75 / 15 / 20 |
| niah_multivalue | 72.5 | 77.5 | 42.5 |
| niah_multiquery | 86.2 | 86.2 | 65.0 |
| vt | 86 | 82 | 74 |
| cwe / fwe | 1 / 80 | 0.5 / 80 | 1 / 76.7 |
| qa_1 / qa_2 | 55 / 30 | 60 / 30 | 55 / 25 |
| **mean** | **64.7** | **63.6** | **57.2** |
| Δ vs dense [95% CI] | — | **−1.1 [−4.3, +2.0]** | −7.4 [−11.0, −4.1] |

- quest **passes** the gate; mv fails it.
- Dense outputs are token-identical across the three RULER runs (260/260).
- quest's remaining losses: −10 on multikey_2 and multikey_3 (2 of 20 prompts each), −4 on vt.

### Qwen3-8B and Llama-3.2-3B, 128k, quest

Same tasks and gate, at the speed-table budgets, on the released build (e7059e52a). Pre-registered with a prediction and with the consequence for this README fixed in advance.

| model (KV cache) | budget | dense | quest | Δ [95% CI] | gate |
|---|---|---|---|---|---|
| Qwen3-8B-Base, YaRN to 128k (q8_0) | 27648 | 59.7 | 60.2 | +0.4 [−2.3, +3.3] | **pass** |
| Llama-3.2-3B (f16) | 28672 | 68.2 | 63.7 | −4.5 [−7.1, −1.9] | **fail** |

- Both were predicted to pass. The 3B prediction was wrong; Qwen's Δ landed just above its predicted interval [−4, 0].
- The 3B loses on multi-needle tasks: multikey_2 −30 (6 of 20 prompts net), multivalue −12.5, multiquery −10, and fwe −6.7. Single needles hold.
- Read after the data, not pre-registered: on multikey_2, all 7 lost prompts are near-misses, a number sharing its first 4–6 digits with the answer (5149787 → 5149780). In 3 of the 20 prompts the answer crosses a page edge, and all 3 are among the losses; the other 4 losses do not cross one. Causality is untested. Per-prompt offsets: `results/cuda_4090_portrq/s_posthoc.txt`.
- Per-task scores: `results/cuda_4090_portrq/`.

### Qwen3-30B-A3B (MoE), 128k, quest

The first MoE model: 48 layers, 32 query and 4 KV heads, 128 experts with 8 active. Same tasks, gate and build; RTX 4090 48 GB; budget 27648. Pre-registered with a smoke test, predictions and the consequence for this README (`protocols/port-mq.md`).

| model (KV cache) | budget | dense | quest | Δ [95% CI] | gate |
|---|---|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 (q8_0) | 27648 | 87.2 | 86.6 | −0.6 [−1.9, +0.4] | **pass** |
| Qwen3-30B-A3B-Instruct-2507, Q4_K_M weights, RTX 5090 (q8_0) | 27648 | 87.8 | 86.8 | −1.0 [−2.2, −0.02] | **pass** |

- The quality prediction held (pass, Δ in [−3, +1]). Its second part, that any loss would sit in multi-needle tasks, is ambiguous: one prompt was lost in multikey_3 and one in qa_2. The speed prediction did not hold (see Speed).
- Losses: multikey_3 −5 and qa_2 −5 (one prompt each); fwe +1.7. Every other task is unchanged.
- The prompts are byte-identical to the Qwen3-8B run's (same tokenizer, same RULER seed).
- The first attempt could not download `tokenizer.json` on the box; the same revision's file (checked against the hub's SHA-256) was copied in and the run restarted (`status_attempt1.txt`).
- Per-task scores: `results/cuda_4090_48gb_portmq/`.
- **Q4_K_M on an RTX 5090** (`protocols/port-mq4.md`, pre-registered with the same gate): passes, as predicted (Δ in [−3, +1]). Losses: qa_1 −5 and qa_2 −5 (one prompt each), fwe −1.7, multiquery −1.25; cwe +0.5. The upper end of the CI is just below zero, so this small loss is distinguishable from no change at 20 prompts per task. Same prompts as the Q8_0 run. Per-task scores: `results/cuda_5090_portmq4/`.

### Where the loss is: multi-needle retrieval

Mean over task groups, dense → quest:

| model | single needle (3 tasks) | multi-needle (5 tasks) | other (5 tasks) |
|---|---|---|---|
| Llama-3.1-8B | 96.7 → 96.7 | 59.8 → 56.8 (−3.0) | +0.1 |
| Qwen3-8B | 93.3 → 93.3 | 49.0 → 49.2 (+0.2) | +0.9 |
| Llama-3.2-3B | 98.3 → 98.3 | 74.8 → 63.2 (−11.5) | −0.1 |
| Qwen3-30B-A3B | 100.0 → 100.0 | 98.0 → 97.0 (−1.0) | −0.7 |
| Qwen3-30B-A3B, Q4_K_M | 100.0 → 100.0 | 100.0 → 99.8 (−0.2) | −2.2 |

- Finding a single fact loses nothing. Retrieving several similar keys or values is the weak spot. The other tasks hold on average, though single tasks move by up to 10 points (2 prompts) either way.
- On Llama-3.1-8B part of the weakness is hidden: dense already scores only 25–30 on multikey_2 and multikey_3. Where both models start high (multivalue, multiquery), the 8B loses nothing and the 3B loses 10–12.5.
- This is a known weakness of page-level selection on synthetic retrieval [3, 4].

### Why mv failed

On the 120 prompts of the six tasks that lost points with mv (niah_multikey_1–3, niah_multivalue, niah_multiquery and vt), each candidate cause was tested separately:

| change | effect on the six-task mean [95% CI] |
|---|---|
| **page score quest instead of mv** | **+12.4 [+6.3, +18.8]** |
| first 2 layers dense | −2.8 [−6.0, 0.0] |
| f32 page statistics instead of f16 | −1.7 [−4.2, +0.2] |

Of the three changes, only the page score helped. A likely reading, not tested directly: a page's mean dilutes a single relevant key among 64, while quest's per-dimension bound keeps it. With mv, larger budgets also pass, at a lower speed: −1.9 [−4.5, +0.6] at 49152 cells (1.23×) and −1.3 [−3.4, +0.7] at 65536 (1.11×), RTX 4090.

### Cheaper checks did not predict RULER

The budgets were first chosen with mv for a mean next-token **KL divergence** of about 0.02 between the sparse and dense output distributions (4 PG19 books, 256 teacher-forced tokens each), plus synthetic needle trials:

| check (mv) | Llama-3.2-3B | Llama-3.1-8B | Qwen3-8B |
|---|---|---|---|
| KL at 32768 cells, 128k | 0.018 | 0.017 | 0.025 |
| Top-1 token changed vs dense, 32768 cells | 4.5% | 6.2% | 10.0% |
| KL / top-1 change at 27648 cells | — | — | 0.031 / 11.3% |
| Needle retrieval at the speed budget, 128k (single / multikey) | 100% / 94% | 100% / 100% | 89% / 100% |

With mv on Llama-3.1-8B these checks passed and RULER did not (KL 0.02, RULER −7.4). Separately, the 3B failed RULER with quest after passing these checks with mv. KL against dense and single needles are not sufficient. Test multi-needle tasks.

## Longer context: 256k

UltraLong-8B (`nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`, Q8_0, KV q8_0), RTX 4090 48 GB, quest.

| context | budget | dense tok/s | sparse tok/s | speedup |
|---|---|---|---|---|
| 128k | 49152 | 40.5 | 47.1 | 1.16× |
| 256k | 49152 | 23.0 | 42.8 | **1.86×** |
| 256k | 98304 | 23.0 | 33.4 | 1.45× |

The same budget gives a larger speedup at a longer context, because dense decode slows with the cache while the sparse read stays fixed.

**Does quality hold at a fixed budget?** Pre-registered test on six RULER tasks (niah_multikey_1–3, niah_multivalue, niah_multiquery, vt): Δ(256k) − Δ(128k) ≥ −3 points at 49152 cells.

| context | dense | quest:49152 | Δ [95% CI] | quest:98304 | Δ [95% CI] |
|---|---|---|---|---|---|
| 128k | 82.4 | 75.5 | −6.9 [−12.0, −2.2] | — | — |
| 256k | 67.4 | 63.4 | −4.0 [−8.7, +0.3] | 63.4 | −4.0 [−7.9, −0.2] |

- The point estimate passes: Δ(256k) − Δ(128k) = +2.9. Its 95% CI [−3.7, +9.7] still allows a loss slightly larger than the 3-point threshold.
- Dense itself loses 15 points from 128k to 256k, so there is less to lose at 256k. This does not show that sparse decoding improves with context.
- Doubling the budget to 98304 cells (the same fraction of the cache as 49152 at 128k) did not change the mean. The loss sits in a few multi-key prompts.
- This six-task mean contains only the harder tasks (those that lost points with mv on Llama-3.1-8B) and is not comparable to the 13-task means above.

KL against dense at a fixed budget rises with context (49152 cells: 0.017 at 128k, 0.023 at 256k; 98304 cells: 0.006 and 0.013).

## The speed–quality dial

Decode speedup at 128k on the RTX 3090 (300 W) by budget, mv score (cells per KV head per step):

| budget | 8192 | 16384 | 27–28k | 32768 | 49152 |
|---|---|---|---|---|---|
| Llama-3.2-3B | 1.75× | 1.56× | 1.34× | 1.27× | 1.07× |
| Llama-3.1-8B | 2.25× | 2.01× | 1.76× | 1.67× | 1.42× |
| Qwen3-8B | 2.30× | 2.07× | 1.80× | 1.70× | 1.43× |

For the 8B, KL against dense (mv) is about 0.11, 0.05, 0.02 and 0.017 at 8192, 16384, 27648 and 32768 cells. RULER was measured only at 27648 (quest and mv), 49152 and 65536 (mv): smaller budgets are faster but their quality is unmeasured.

## How the speedup is predicted

Time per decoded token at context length T and budget B:

```
t_sparse(T) = t_dense(T) − L · [a_dense(T) − a_sparse(T, B)] + o0 + o1·T
```

- `t_dense(T)`: measured dense decode time per token at the target length.
- `a_dense(T)`, `a_sparse(T, B)`: per-layer attention kernel time, dense and sparse, from microbenchmarks at the target length.
- `L`: the number of layers.
- Times are per decoded token, in ms.
- `o0 + o1·T`: the cost of selecting pages, fitted on the target GPU at 8k–32k. This is the only extrapolated term.

On the RTX 3090 (300 W, mv) the predictions were committed before the 64k and 128k sparse runs. All 6 landed within ±10% of measurement; the worst miss was 6.2%.

At 256k (8× beyond calibration, RTX 4090 48 GB) the model failed its ±10% gate: 5 of 8 points within 10%. It over-predicted at 128k by 5–7% and under-predicted at 256k by 7–15%, so it is conservative where the gain is largest.

## What did not work

The gated experiments were pre-registered (see [protocols/](protocols/)); analyses made after the data are labelled as such. Besides the quality results above (mv's −7.4, the 3B's failure, the 256k cost-model miss), these failed or had to be fixed:

- **The first CUDA build gave only 1.05–1.18× at 128k.** Selection overhead was the cause: two ggml kernels (`pool_1d` with a whole-row window, and `top_k_mask`) used too few GPU threads. Fixing them gave the numbers above. The `pool_1d` fix is proposed upstream as [ggml-org/llama.cpp#31855](https://github.com/ggml-org/llama.cpp/pull/31855); the `top_k_mask` fix stays in the fork for now and will be proposed with the feature.
- **The `top_k_mask` fix first made prefill-sized batches up to 2.2× slower.** The final version keeps master's kernel when rows fill the GPU: decode cases at a 128k cache are 1.9–2.0× faster and no case is more than 0.6% slower.
- **One kernel microbenchmark was mis-designed.** It picked pages independently per head, while the engine shares pages across a GQA group, so its negative verdict does not apply to the engine. It is kept in `protocols/`.
- **Qwen3-8B at the Llama budget narrowly failed its first quality gates** (mv): KL 0.031 against a 0.03 limit, and single-key retrieval at 89% against 90%. With quest it later passed RULER at that budget.
- **The MoE speedup missed its prediction**: 1.41× at 128k against a predicted 1.5–2.3× (Qwen3-30B-A3B, RTX 4090 48 GB). Read after the data: the protocol assumed the KV read was about 2/3 of the MoE's dense time; it is about 80%, so the miss comes from what the prediction left out, the sparse step's own overhead across 48 layers (see Speed). At Q4_K_M on an RTX 5090 the speed prediction held (1.72×).
- **The 8B in-engine KL was 30% above the Python reference** (gate ≤ 25%; the 3B was within 5%). The likely causes are q8_0 weights and TF32 in cuBLAS scoring; this is not resolved.

## Scope and limits

- Batch size 1, decode only, CUDA only. Prefill is unchanged and dominates wall-clock at 128k.
- Four GPU types (two RTX 3090 cards, RTX 4090 24 GB, RTX 4090 48 GB, RTX 5090), two model families (Llama 3.x, Qwen3): dense 3B–8B and one MoE (Qwen3-30B-A3B, about 3B active, at Q8_0 and Q4_K_M).
- Quality: RULER at 128k on four models (the MoE at two quantizations), 20 prompts per task, one run per configuration; six RULER tasks at 128k and 256k on UltraLong-8B; KL on 4 PG19 books; 40 synthetic needle trials per model.
- RULER is synthetic. Real long-document tasks are covered only by its two QA tasks.
- Multi-needle retrieval loses points, on small models most (see Quality).
- Sparse decode output is not bitwise reproducible from run to run (dense is). Two quest runs of 30 RULER prompts on the same build gave token-identical outputs on 21 and 22 of 30, and their mean scores differed by 0.8 points (53.5 and 52.7). The released branch matches the research build within that spread (21–24 of 30).

## Repository

- [`protocols/`](protocols/): the pre-registrations, with the commit and time each was committed.
- [`results/`](results/): raw results and analyses, one folder per experiment; [`results/README.md`](results/README.md) maps each folder to its protocol, hardware and README section.
- [`scripts/`](scripts/): `scripts/reproduce.sh` recomputes the speed tables, RULER and cost-model numbers in this README (not the rough MoE breakdown) from `results/` (no GPU needed), and the scripts to rerun the experiments. See [`scripts/README.md`](scripts/README.md).
- The released fork commit e7059e52a is 5cf439778, checked against the research build in `results/cuda_4090_48gb_featcheck/`, plus the CUDA-only guard.

## References

1. Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, Song Han. *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)
2. Dich Nhat Minh Nguyen, Tran Dang Duong Nguyen. *MOIRA: Mass-Oriented Indexing with Ragged Attention for Long-Context Decoding.* 2026. [arXiv:2610.04313](https://arxiv.org/abs/2610.04313)
3. Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, Edoardo M. Ponti. *The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs.* Findings of ACL 2026. [arXiv:2504.17768](https://arxiv.org/abs/2504.17768) (§4.2: Quest degrades on synthetic RULER NIAH; page granularity amplifies it.)
4. Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, Lili Qiu. *RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval.* 2024. [arXiv:2409.10516](https://arxiv.org/abs/2409.10516) (Table 9: Quest at 2K tokens, 128k RULER, keeps single needles and loses on multi-key, multi-query and multi-value.)

## Licence

Text, protocols and results: [CC BY 4.0](LICENSE). Scripts: [MIT](LICENSE-MIT). The llama.cpp fork is MIT, as llama.cpp.

---

Giorgio Sarno, Controra Research
