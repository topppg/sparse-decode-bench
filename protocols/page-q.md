<!-- first committed in the project repository as cc426c7 at 2026-11-03T21:05:40+01:00 -->

## PAGE-Q (2026-11-03): does page selection keep quality at a budget the Metal kernel accelerates? (pre-registered before any PAGE-Q data; one smoke run seen, Qwen2.5-0.5B at 4k, 1 book)
**Question.** Two parts:
- What cell budget does a 64-cell page selector need for mean next-token KL 0.02 at 32k–128k, compared with kq2 (R-PREDICT-128K)?
- What projected decode speedup does the K2 V2 kernel give at that budget?

**Setup.**
- `t4_select.py`, P-T4 protocol (dense prefill, sparse 256-row teacher-forced tail).
- fp32 with `--tf32`, as in R-PREDICT-128K. The same 4 PG19 books: same seed, n-seqs and maximum length 131072.
- **Models:** Llama-3.1-8B (primary) and Llama-3.2-3B (secondary).
- **T** ∈ {32768, 65536, 131072}.
- **Modes, all GQA-shared over the group, sink 16 + window 64 always kept:**
  - `mv64` (primary): page score s·μ + ½ Σ s_d² σ_d², with s = scaled query and μ, σ² the per-channel page mean and variance (a Gaussian moment estimate of page mass, MOIRA-style).
  - `mean64`: s·μ.
  - `quest64`: Quest min/max bound.
- **Budgets** (cells including sink and window): 512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152.
- **Retrieval at 131072:** single and multikey, 4 books, modes as above. k_ret is defined as in R-PREDICT-128K.

**Estimands.**
- `r*_page(T)`: the read_star at ε = 0.02 (mean cells read per KV head).
- The ratio `r*_page / r*_kq2`, using kq2 from `results/pred_llama31_8b`.
- `S_proj(T)`: S from `results/port_k2_v2/analysis.json` (8B layout), interpolated linearly in B at that T, evaluated at B = r*_page(T). If r* > 32768, S_proj is undefined and treated as a failure.

**Decision (Llama-3.1-8B, mv64):**
- **GO** for llama.cpp integration if S_proj(131072) ≥ 2.0 **and** k_ret(mv64, 131072) ≤ r*_page(131072) for both single and multikey.
- **NO-GO** otherwise.
- Reported, not used for the decision: S_proj at 32k and 64k, the other modes, the 3B model, and the 99th-percentile KL and flip tails at the chosen budget.

**Hypotheses:**
- **PQ1:** r*_page(131072) ≤ 20480, the MOIRA-like 15.6% kept.
- **PQ2:** r*_page / r*_kq2 ≤ 3 at every T.
- **PQ3:** mv64 ≤ mean64 ≤ quest64 in r* at 131072.
