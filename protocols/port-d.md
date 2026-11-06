<!-- first committed in the project repository as e40f699 at 2026-11-06T10:57:55+01:00 -->

## PORT-D (2026-11-06): why RULER fails at the KL budget — score, early layers, or stats precision (pre-registered before any PORT-D data; R1 outputs analysed as below; R4 running, not read; PORT-L Phase 1 running)

**What R1 already shows (post hoc, from r1_gen.jsonl).** In the six losing tasks, sparse output first differs from dense at output token 2–3 (the first answer digit). In niah_multivalue, sparse loses 25 of the 58 values dense finds and writes 101 wrong numbers, of which only 32 occur in the prompt. In niah_multiquery, values are omitted (17 lost, 3 wrong numbers). In multikey_3, all 4 wrong numbers occur in the prompt. Reading: the error is made at the answer token, and mostly because the needle page was not selected, not because of drift over a long answer.

**Candidate causes.** (S) the mv score dilutes a single outlier key in a 64-cell page, where Quest's per-dimension bound does not; (E) every layer is sparse, while Quest-style methods keep the first two dense; (P) RULER used f16 page stats, while every earlier quality gate used f32.

**Setup.**
- Llama-3.1-8B q8_0 (SHA-256 must match the PORT-G/R file 38a8de72…; mismatch reported), KV q8_0.
- Prompts: the 120 R1 prompts of niah_multikey_1–3, niah_multivalue, niah_multiquery and vt, extracted unchanged from `ruler8b.i32` (subset SHA-256 prefix f145558a4f91b4d4). Scoring as R1 (`port_r_score.py`).
- Budget 27648 cells throughout (the R1 budget).
- D-A on `vastai14` (RTX 4090 48 GB), llama.cpp `sparse-cuda` 5242a9188 (adds `LLAMA_SPARSE_DENSE_LAYERS`; a config suffix `:dN` keeps layers < N dense), stats f16, one process: dense, mv:27648, quest:27648, mv:27648:d2, quest:27648:d2.
- D-B on `vastai11` (RTX 4090 24 GB, the R1 box and build 8213b023d), stats f32: dense, mv:27648.

**Measures.** Δ(config) = config − dense on the six-task mean, paired bootstrap as R1. Each effect is the difference of two Δs over the same 120 prompts (paired bootstrap of the per-prompt difference).
- S: Δ(quest) − Δ(mv). E: Δ(mv:d2) − Δ(mv). P: Δ(mv, f32, D-B) − Δ(mv, f16, R1 outputs on these 120 prompts).
- An effect **counts** if it is ≥ +3 points and its 95% lower bound is > 0.
- Check: dense outputs of D-A and D-B against R1 dense, token by token (reported).

**Prediction.** S counts; E is positive but below +3; P is within ±2. quest:d2 is the best config.

**Consequence for PORT-L (fixed now).** If S or E counts, PORT-L L2 and L3 use the best of {mv, quest, mv:d2, quest:d2} by Δ, at the budgets fixed by the PORT-L rule; otherwise mv as registered. Recorded as a PORT-L amendment before any L2/L3 data. R4 is not rerun.

Order: D-A after PORT-L Phase 2 on `vastai14`; D-B after PORT-R2 R4 and U3b on `vastai11`.
