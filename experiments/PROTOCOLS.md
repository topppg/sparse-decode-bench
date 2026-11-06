# Protocols — programme A ("is there a saddle?")

All scripts are in `src/` and write to `results/<tag>/`. Logs are in `logs/`. Seeds are fixed in the code.
Data: WikiText-103 (validation split for probes, train split for A4-fine).
Rationale and decisions are in `../LOGBOOK.md` (D-xxx).

Common definitions:
- **Item:** a prefix `x = tokens[0..t]` and target `y = tokens[t+1]`, with `t ∈ [32, 255]` (Pythia) or `[16, 126]` with a BOS token (GPT-2).
- **Edge:** `(layer, head, query i, key j)` with `j ≤ i`. `E = L·H·T(T+1)/2`.
- **Attribution:** `c_e = a_ij · ∂log p(y|x)/∂a_ij` (D-004).
- **f90:** the smallest fraction of edges whose |c| sums to ≥ 90% of Σ|c|. `n90` is the same as a count.
- **Cancellation:** `1 − |Σ c| / Σ |c|`. 0 means all contributions share a sign; near 1 means strong interference.
- **Budgets (D-012):** every selector keeps the same number of edges per head. Rows may be empty; an empty row outputs zero.
- **Null overlap:** empirical, from random global and random row-stratified sets of equal size. Not assumed.

## A1 — concentration and oracle fidelity (H1)
`src/a1_a3_final.py`. Pythia-160M at step143000, 64 sequences × 4 targets = 256 items.
- **Measures:** f50/f90/f99, participation ratio and cancellation of `c`; attention effective keys per layer; sink mass. Oracle-mask fidelity (D-005) at budgets 0.5–20% for six selectors.
- **Supports H1:** median f90 ≪ 1 (say < 5%), **and** attribution masks reach median KL < 0.05 nats at a budget ≤ 5% while random and local masks don't.
- **Refutes H1:** attribution masks are no better than attention or local masks, or fidelity needs > 20% of edges.
- **Also reported:** Spearman correlation between f90 and predictive entropy (is the saddle sharper when the model is confident?).

## A3 — out-boundary dependence (H3)
Same run as A1.
- **Measures:** Jaccard and mass-recall between the top-k edge sets (1%, 5%) for `y` and for `y' ∈ {alt1, rand}`, for both `log p` and raw-logit objectives. Cross-fidelity: the mask from attribution on `alt1`, used to recompute `log p(y)`.
- **Reference points:** the attribution-vs-attention overlap (a router that imitates attention), and chance Jaccard.
- **Supports H3:** raw-logit overlap between `y` and `y'` close to chance, **and** a material fidelity gap between the `attr_y` and `attr_alt1` masks.
- **Refutes H3:** `y`/`y'` overlap ≈ 1 and the fidelity of `attr_alt1` ≈ `attr_y`. The in-boundary alone then fixes the saddle.
- **Leakage (D-014):** the fidelity gap on log p(y) between attr_y masks and counterfactual-label masks (attr_alt1, attr_rand). A large gap means the mask carries y-specific information, so it must never be used in a training forward pass.

## A2 — basis dependence (H2)
`src/a2_basis.py`. GPT-2 small, residual SAEs at resid_pre of layers 2, 6 and 10, 128 items, context ≤ 128 (the SAE training length).
- **Measures:** n50/n90/n99 of the attribution of `log p(y|x)` over residual-stream nodes, in six decompositions: raw, random-rotated, PCA, SAE-collapsed (reference only), SAE-atomic, and random-dictionary-atomic. Also SAE FVU, L0, atomic error share and completeness gaps. Cancellation is reported but is confounded by LayerNorm scale invariance (R-A2 v1, caveat 3).
- **Bookkeeping (D-013):** atomic nodes in every basis. Controls: PCA (an invertible learned basis) and a random dictionary at matched L0.
- **Supports H2:** SAE-atomic n90 well below raw, PCA **and** random-dictionary n90, with atomic error share < 20%. Note: node counting penalises decompositions whose terms cancel, so H2 needs the intervention-fidelity follow-up (TODO) before any positive claim.
- **Refutes H2:** comparable n90, or a large SAE advantage that lives mostly in the error node.
- **Control:** if raw ≈ rotated, the neuron basis isn't privileged, and the comparison is between "a generic basis" and "a learned sparse basis".

## A4-ckpt — stability across training checkpoints (H4, coarse)
`src/a4_checkpoints.py`. Pythia-160M at 14 checkpoints (step 0 … 143000), 64 fixed items.
- **Measures, per checkpoint:** loss on the items, the concentration metrics, attention effective-key fraction per layer, and fidelity (KL) of the `attr_y`, `attn_row` and `local_sink` masks at 1% and 5%. **Stability:** Jaccard of top-k sets (attribution, and per-row attention) between consecutive checkpoints and against the final checkpoint.
- **Annealing picture (Hopfield regimes):** effective keys should drop from ≈ full rows at step 0 toward a few keys. When it happens per layer is the observable.
- **Stale-mask transfer:** the mask from checkpoint s is applied to checkpoint s′. KL against the fresh mask, plus mass recall. This is the operational question ("can a stale saddle be reused?"). Overlaps are reported with Δstep and log(s′/s), because gaps are uneven.
- **Supports H4:** transfer KL close to fresh-mask KL, with mass recall > 0.5, at gaps of the order of 10³ steps late in training.

## A4-fine — step-level stability while training (H4, fine)
`src/a4_train_tiny.py`. A 29M parameter GPT-NeoX trained from scratch, 6000 steps, probes every 20 steps.
- **Probes** every 20 steps, plus every step inside 5 dense windows (40 steps at 100, 400, 1000, 2000, 4000).
- **Measures:** train-loss EMA, probe loss, induction score and control (marker = difference-in-differences), f90, effective-key fraction per layer, Jaccard to the previous probe (attribution and attention, 1%), and stale-mask KL against fresh-mask KL with mass recall. At the end: median Jaccard at lags 1/5/20/100/500/1000/2000 (from step 100 on), and Jaccard to final over time.
- **Supports H4:** stale-mask KL ≈ fresh-mask KL at lag 20, high Jaccard at lags 1–20 that decays slowly, with dips aligned to the rise of the induction marker.
- **Refutes H4:** lag-20 Jaccard is low outside transitions, i.e. the saddle jumps on the timescale of single optimizer steps.

## Known limitations (all of A)
- Small models, short context. Sparsity is expected to matter more at long context, which isn't tested.
- Attribution is first-order. Fidelity recomputation is exact but uses an *oracle* (it needs the dense pass to pick the edges). A measures whether a saddle exists, not whether it can be found cheaply.
- A single dataset (WikiText-103).

---

# Programme B — can a *non-oracle* "where" train as well as dense? (GPU, DRAFT v0, pending Codex review)

**Question.** Programme A showed the relevant set exists and is stable, but every selector was an oracle. B asks whether a **cheap**
selector, computed without the dense attention matrix, can train a model to (near) dense quality at a fraction of the attention
compute, and whether the spin-foam-inspired levers help: temperature-adaptive budgets (C1) and amortised refresh (C2).

**Key correction from A (D-017).** A4 showed stability for the *same inputs* across training time. A training batch is *new data* every
step, so a stale content-specific mask is meaningless. What can be amortised is the **per-head structure** (budget allocation,
regime/temperature). Content-specific selection must come from a cheap **per-input** scorer.

**Setup.** GPT-NeoX-style decoder, ~125M parameters (12 layers, d=768, 12 heads); context 4096; FlexAttention with 128×128 blocks;
bf16; a FineWeb-Edu sample (~500M tokens, a fixed shuffled stream, identical across arms); the same seed, optimizer and schedule for every arm.

**Arms (equal attention-block budget b ∈ {12.5%, 25%} of causal blocks, except dense).**
1. **Dense** (causal FlexAttention): the reference.
2. **Window + sink** (static; fixed per-row block budget): the StreamingLLM-style baseline.
3. **Block top-k, uniform k per head** (cheap scorer = mean-pooled q-block × mean-pooled k-block, MoBA/NSA-like; the local block and the sink block are always kept): the existing-method baseline.
4. **Block top-k, temperature-adaptive k per head (C1 + C2).** Same scorer as 3. The per-head budget is allocated from each head's measured
   effective-key fraction on a small dense probe batch, refreshed every R steps (R ∈ {100, 1000}). Total budget equals arm 3.
5. **Oracle block mask** (global top-k of dense attention block mass at the same budget). An upper bound for the selector gap; costs a dense pass.

**Measures.** Validation loss (dense evaluation and in-mode evaluation) against tokens and against attention FLOPs (counted from kept blocks);
wall-clock with FlexAttention; per-head budget trajectories (does the C1 allocation track the annealing seen in A4?); selector recall against
the oracle (block mass captured).

**Pre-registered outcomes.**
- **C1 supported:** arm 4 closes ≥ 30% of the arm-3-to-dense validation-loss gap at equal b, at both budgets.
- **"Cheap where" viable:** arm 4 at b = 25% ends within 0.02 nats of dense.
- **C2:** R = 1000 within 0.005 nats of R = 100. Otherwise amortisation fails at this timescale.
- **Selector gap:** (arm 3 or 4) − arm 5 measures how much a better scorer could still gain.

**Phase 0 on the GPU (before any paid long run):** a parity check that the Mac A1 numbers reproduce on CUDA within tolerance; FlexAttention
correctness (a dense BlockMask matches SDPA); throughput measured per arm; a 20M-token pilot per arm to size the full run.

**Known risks.** Early-training instability under sparsity (A4: the transition phase) is mitigated by a dense warm-up of W steps, logged as an
explicit decision. Mean-pooled scorers miss isolated needles. Consumer-GPU host interruptions are mitigated by checkpointing every N steps.

---

# Programme B v1 — token-sparse training with a learned indexer (supersedes B v0; follows R-B0)

**Question.** With the "where" found by a cheap *learned* indexer at token granularity, does training under 5% token sparsity keep dense quality? And do temperature-adaptive per-head budgets (C1b), amortised over R steps (C2), beat uniform per-query budgets (DSA-like)?

**Script:** `src/b1_train.py`.

**Model.** GPT-NeoX, 6 layers, d=512, 8 heads (d_h = 64), FFN 2048, about 70M parameters (embedding-heavy). Context 2048. Pythia tokenizer. WikiText-103 train. bf16 autocast. AdamW (lr 1e-3, 100-step warm-up, cosine to 10%). Batch 8 × 2048.

**Protocol.**
1. A shared **dense warm-up of W = 300 steps**. The indexer trains throughout, by KL to dense attention.
2. Branch from the identical state (model, indexer, optimizers, scheduler, and the same data order) into three arms:
   - **dense**;
   - **dsa**: uniform k = 52 keys per query (≈ 5% of causal pairs), selected by the indexer;
   - **adaptive**: per-head k_h from greedy equal-marginal-mass allocation over per-head top-k mass curves measured on the training batch every R = 100 steps; Σk_h = L·H·52.
3. The indexer keeps training in every arm.

**Indexer.** Per head, rank 16: (q·A_h)(k·B_h)ᵀ, on *detached* q and k. Cost ≈ 16/64 of the QKᵀ score FLOPs.

**Measures.**
- validation CE under dense evaluation and under in-mode (sparse) evaluation, every 100 steps;
- train loss;
- indexer KL;
- the kept-pair fraction (measured);
- the k_h trajectories.

**Pilot (1 seed).** Checks that it trains, sizes the effects, and estimates step-to-step noise. **Confirmatory (3 paired seeds)** only if the pilot shows a gap of at least about 0.02 nats between any two arms.

**Honest accounting.**
- The quality implementation computes dense attention and then masks it, so there is **no wall-clock saving**. FLOPs are counted analytically: sparse attention = kept fraction × dense attention FLOPs, plus indexer FLOPs, plus one dense probe per R steps (adaptive arm), plus the W dense warm-up steps.
- The indexer target is *dense* attention, which is optimistic. DSA restricts the target to the selected tokens. A real deployment would need that, or a periodic dense probe.

**Pre-registered reading (pilot).**
- Sparse ≈ dense (|Δ| < 0.02 nats, in-mode evaluation) supports "token-sparse training with a learned where is viable".
- adaptive < dsa by > 0.01 is a C1b signal worth confirming.

---

# Programme H — sum over histories vs mean-field (user request, 2026-10-29)

**Question.** The real transformer is a mean-field closure: the next layer's geometry α is built from the *averaged* state. A genuine sum over histories (stochastic hard attention, where every query samples one key and each branch builds its own geometry) is a different model. How large is the gap, where does it come from, and can a model be trained as a sum over histories?

## H0 — frozen-model decomposition of the mean-field gap (`src/h0_histories.py`)
Pythia-160M, 8 sequences × 512 tokens, S = 64 branches per sequence. In each branch, every (layer, head, query) samples one key with Gumbel-max from α. The output is averaged over branches.

| variant | α | LN scales r | MLP gates χ | gap measures |
|---|---|---|---|---|
| V0 | frozen (soft pass) | frozen | frozen | **Should equal soft up to Monte-Carlo noise.** Numerical check of Prop. 2 (the network is multi-affine in independent one-hot selections) |
| V1 | frozen | frozen | own | MLP non-linearity |
| V2 | frozen | own | own | + LN non-linearity |
| V3 | own (per branch) | own | own | + loops (a branch-dependent geometry): the true sum over histories |

Plus a layer sweep: V3 sampling at one layer only (12 runs), and V0 at layers {0, 6, 11}.

**Metrics:**
- relative error of the branch-mean logits vs soft logits (centred), against the Monte-Carlo standard error;
- KL(soft ‖ softmax(mean logits));
- KL(soft ‖ mixture E_σ softmax(z_σ));
- NLL of the soft model vs the mixture.

**Reading:**
- V0 ≈ MC noise validates the state-sum identity.
- V1 → V3 increments attribute the mean-field gap to MLP, LN and loop effects.
- Caveat: the model was trained soft, so V3 degradation is expected. H0 measures the *structure* of the gap, not trainability.

## H1 — training as a sum over histories (planned after H0)
Small GPT-NeoX (as in B1), trained with stochastic hard attention (straight-through / Gumbel estimator; 1 or K branches) against soft at equal tokens. Evaluated by the mixture likelihood. Protocol to be written after H0.

---

# R1 — the lock-in result (pre-registered 2026-10-29, before any R1 run)

**Claim.** A decoder trained with token-sparse attention, where each query attends to only **k = 10 keys (≈ 1% of causal pairs)** selected by a rank-16 learned indexer **supervised only on the keys it selected** (no dense attention after the shared warm-up), reaches the validation loss of dense training at equal tokens.

**Setup.** `src/b1_train.py` as in B1 (70M GPT-NeoX, 6 × 512, 8 heads, context 2048, WikiText-103, batch 8, lr 1e-3 cosine). 300 dense warm-up steps (the indexer trains by dense KL, as in DSA's warm-up stage), then 900 branch steps per arm, branched from the identical state.
- **Arms:**
  - `dense`;
  - `dsa_sparse` (**the claim**): uniform k = 10, indexer KL restricted to the selected keys, targets = sparse attention probabilities;
  - `dsa` (reference): k = 10, indexer KL to dense attention.
- **Seeds:** 0, 1, 2 (each seed changes the init, the data order and the warm-up).
- **Validation:** 96 sequences × 2048 tokens (≈ 197k tokens), fixed.
- **Eval:** in-mode, every 100 updates.

**Primary metric.** Δ = L(dsa_sparse) − L(dense), the final in-mode validation loss, paired per seed.
**Decision rule.** The claim holds iff the one-sided 95% upper bound (t, df = 2) of the mean Δ is **< 0.02 nats**.
**Secondary metrics.** The last-3-eval mean, the dsa reference arm, dense-evaluation of the sparse models, and the kept fraction.

**Honest accounting** (reported, `src/r1_analyze.py`):
- this quality implementation computes dense logits then masks, so **no wall-clock saving is claimed**;
- analytic FLOPs: sparse attention = kept × dense attention, plus indexer (O(T²·r) scores + projections);
- the dense warm-up is 25% of the steps.
**Scope of the claim:** 70M parameters, 2k context, 14.75M tokens per arm in the branch (900 × 8 × 2048) plus 4.9M shared warm-up tokens. (Corrected per REV-005 #3; the decision rule is unchanged.)

---

# Parallel gates on GPU 2 (A100 80 GB, `vastai2`), pre-registered 2026-10-29

## G-SIZE — is the input-dependent saddle size robust to model scale? (extends R-A1c)
Rerun `a1_a3_final.py` unchanged (same 256 items, budgets, selectors) on **Pythia-410M and Pythia-1B** (step143000). Then apply the R-A1c analysis (min budget for KL < 0.05 vs predictive entropy).
- **Holds:** Spearman(min budget, entropy) ≥ 0.4 for both, and the easiest quartile's median min budget ≤ ½ of the hardest quartile's (oracle selector), for both.
- **Also reported:** whether the overall f90 and fidelity curves shift with scale.

## G-C3 — difficulty-adaptive per-query budgets (frozen Pythia-160M, 2048 context, `src/b2_gates.py`)
At equal total kept pairs (kept_frac is checked per variant), compare fixed-k per query (DSA-like) against:
- **need**: each (layer, head, query) gets keys until it covers a common attention-mass fraction τ, with τ set by bisection so the totals match;
- **entropy**: keys per position ∝ the model's predictive entropy at that position (label-free), shared across heads.

Both with oracle scores (is the lever there?) and with a rank-16 learned indexer (is it cheap?). Budgets 1/2/5%. Metric: mean KL over positions, 32 sequences.
- **Lever present:** oracle|need beats oracle|fixedk by ≥ 1.5× in median KL at ≥ 2 of the 3 budgets.
- **Lever cheap:** indexer|need beats indexer|fixedk at the same budgets.

## G-C2F — coarse-to-fine selection (same script)
Candidate blocks per query come from a cheap scorer (q · mean-pooled block key), keeping the own block plus top-m blocks (5/10/20% of blocks; block sizes 64 and 32). Then token top-k (oracle or indexer) inside the candidates, at 1/2% budgets.
- **Passes:** some filter at ≤ 20% candidates keeps mass recall ≥ 0.9 vs the unfiltered oracle selection, and KL within 1.5× of the unfiltered selection.
- Otherwise, mean-pooled block filters are ruled out as the sub-quadratic "where", and a better block scorer is needed.

## G-C2F2 — better block representatives for coarse-to-fine (T3), pre-registered 2026-10-29 (`src/b3_c2f.py`)
Same setup as G-C2F (frozen Pythia-160M, 2048 context, 32 sequences, oracle token selection inside the candidates, fixed-k at 1% and 2%). Block scorers:
- `mean` (as G-C2F);
- `maxpool` (q · elementwise-max key);
- `bound` (a certified upper bound on the block's max logit, q·c_b + ‖q‖·r_b, with centroid c_b and radius r_b);
- `cent4` (4 k-means centroids per block, max over centroids).

Blocks of 64 and 32 tokens, candidates 10% and 20% of blocks.
**Passes** if some scorer at ≤ 20% candidate *pairs* reaches mass recall ≥ 0.9 against the unfiltered oracle and KL ≤ 1.5× unfiltered.

## T2-SCALE — how the needed keys scale with available keys (T2 contribution, pre-registered 2026-10-29, `src/t2_scaling.py`)
For every (layer, head, query row i), compute n_p(i), the number of top keys covering a fraction p ∈ {0.5, 0.9, 0.99} of that row's attention mass. Also the effective keys exp(H). Average within log-spaced bins of row length i, and fit **n_p(i) ∝ i^γ** over i ≥ 64, per layer.
- Models / context: Pythia-160M, 410M and 1B at 2048 (16 sequences); **Qwen2.5-0.5B at 2048 and 8192** (16 and 8 sequences); WikiText-103 validation; bf16 weights, fp32 softmax.
- **This is a measurement, not a pass/fail.** Hypotheses stated in advance:
  - (a) γ₉₀ < 1 for most layers in every model (the needed keys grow sub-linearly);
  - (b) the direction of the γ trend with model size (160M → 1B) is reported, with no sign predicted;
  - (c) for Qwen, the γ fitted at 8k agrees with 2k within 0.15 (no regime change at longer context).
- **Caveats:** mass coverage is not output fidelity (fidelity at matched k is a later step). Attention sinks can dominate low-p levels.
- **Prior art to check before any claim:** the Sparse Frontier ("longer sequences tolerate higher sparsity", qualitative), and attention-entropy scaling work (e.g. Scalable-Softmax).

## T2-FID — fidelity version of the scaling law (pre-registered 2026-10-29, `src/t2_fidelity.py`)
k*(T) is the smallest fixed number of keys per query (oracle top-k by exact logits, every layer and head) such that the mean KL(dense ‖ top-k) over all positions is < ε, for ε ∈ {0.05, 0.02, 0.01}. It is log-interpolated on the grid k ∈ {4, …, 1024}. Also computed restricted to the last quarter of positions (the long rows). Fit **k* ∝ T^γ_F**.
- Qwen2.5-0.5B at T ∈ {1k, 2k, 4k, 8k}; Pythia-160M/410M/1B at T ∈ {512, 1k, 2k}; 8 sequences each; WikiText-103 validation.
- **Hypotheses stated in advance:**
  - (a) γ_F < γ₉₀ from T2-SCALE (output fidelity needs fewer keys than mass coverage);
  - (b) γ_F < 1;
  - (c) report whether γ_F decreases with model size (as γ₉₀ did).

### T2-FID extension (pre-registered 2026-10-29, queued while the user is away)
The same protocol and hypotheses as T2-FID, adding: (i) **Qwen2.5-0.5B at T = 16384** (4 sequences), to test whether the exponent holds one octave further; (ii) **Qwen2.5-1.5B at T ∈ {1k, 2k, 4k, 8k}** (a model-size axis at long context). fp32.

### T2-FID domain robustness (pre-registered 2026-10-29)
The same k*(T) protocol on **code** (`codeparrot/codeparrot-clean-valid`), Qwen2.5-0.5B and 1.5B, T ∈ {1k, 2k, 4k, 8k}, 8 sequences, fp32.
**Holds** if γ_F(code) is within 0.15 of γ_F(WikiText) for the same model at ε = 0.05 and 0.02. Code has more long-range, exact-copy structure, so a different exponent there would be informative either way.

---

# Round 3, pre-registered 2026-10-29 ("run them all")

## R2 — k(T)-scaled sparse training (R1 pushed forward with T2's law) — `b1_train.py`
R1 setup (70M, WikiText, 300 + 900 steps, sparse-only indexer supervision, seed 0), **16,384 tokens per step at every T**. Validation ≈ 197k tokens (n_val = 196608 // T).
Arms at each T ∈ {512, 1024, 4096} (T = 2048 is R1 seed 0, same configuration):
- `dense`;
- **law**: k = round(10·(T/2048)^0.55) → 5 / 7 / 15;
- **frac** (fixed fraction ≈ 0.49%): 3 / 5 / 20;
- **fixed**: k = 10.

Hypotheses (Δ = in-mode final loss minus dense at the same T):
- (a) law Δ < 0.02 at every T;
- (b) fixed-k degrades with context: Δ_fixed(4k) − Δ_fixed(512) > 0.005;
- (c) at 4k, |Δ_law − Δ_frac| < 0.005 while law uses 25% fewer keys.

Single seed (R1's seed sd ≈ 0.0008 sets the noise scale). Caveat: T2's exponent was measured on frozen models with oracle selection; a trained model with a learned indexer may need a different one, and that difference is itself a result.

## T3-PROP — saddle propagation across layers — `t3_propagation.py`
Frozen Pythia-160M, fp32, T ∈ {512, 1024, 2048}, k ∈ {10, 20}, 8 sequences. Layer 0 searches all keys. Layer ℓ ≥ 1 searches only the candidates C = (union over heads of layer ℓ−1's selected keys for the query) ∪ (for each of those keys, the top-2 keys *it* selected) ∪ (a local window of 16) ∪ (sink positions 0–3). Recursive: each layer's selection feeds the next. Variants: hop0 (no hop), **hop1**, and hop1-random (random candidates, same per-row count).
**Promising** if hop1 recalls ≥ 0.9 of the oracle top-k mass, beats random by ≥ 0.2, **and** the candidates per row grow sub-linearly in T (fitted exponent < 0.5 over 512 → 2048). That would make locating cost ~T·|C| instead of T².
Caveat: layer 0 still searches everything.

## T2-CONS — consolidation — `t2_fidelity.py --bootstrap 1000`, 16 sequences
- SmolLM2-360M (Llama family) on WikiText, code and C4 at 1k–8k;
- Qwen2.5-0.5B on C4, plus WikiText re-run with 16 sequences.

Report the 90% bootstrap CI of γ_F.
- **Holds** if the CI upper bound is < 1 everywhere (sub-linear), and if the domain ordering "code < WikiText" seen on Qwen reappears on SmolLM2.

## T2-MI — theory link — `t2_mi.py`
Bipartite information I(L) = Σ over the second half of [NLL without the first half − NLL with it], for L ∈ {128 … 4096}, 16 sequences. Fit I ∝ L^β. Qwen2.5-0.5B on WikiText, code and C4.
**Exploratory:** does the ordering of β across domains match the ordering of γ_F? Three domains give only a weak test. Report it, don't claim it.

---

# Overnight plan (pre-registered 2026-10-29, before R2's fixed-k = 10 arm at 4k finished)

## T3-PROP v2 — `t3_propagation.py --v2`
Same setup as T3-PROP (Pythia-160M, fp32, T ∈ {512, 1024, 2048}, k ∈ {10, 20}, 8 sequences). Variants:
- hop1 with fanout 4;
- hop2 with fanout 2;
- hop1 (fanout 2) + full search every 3rd layer;
- hop1 (fanout 4) + full search every 3rd layer;
- that last one with random candidates, as a control.

Note: the refresh variants pay full search on 1/3 of the layers (≈ 1/3 of the quadratic cost).
**Improved** if some non-refresh variant reaches recall ≥ 0.9 at T = 2048 with candidates per row still flat in T. Refresh variants are reported with their cost.

## T2-MI on SmolLM2-360M — `t2_mi.py --model HuggingFaceTB/SmolLM2-360M`
WikiText, code and C4. Compare with SmolLM2's γ_F (T2-CONS: 0.63 / 0.66 / 0.60).
**Link supported for this model** if |β − γ_F| ≤ 0.05 in at least 2 of 3 domains.

## R3 — conditional lock-in, decided by R2's fixed-k arm at T = 4096 (rule fixed now)
- If Δ_fixed10(4k) ≤ 0.012 ("**training compresses the saddle**"), run **R3a**: seeds 1 and 2 at T = 1024 (dense, sk10) and T = 4096 (dense, sk10, sk15).
  **Claim:** constant k = 10 keeps dense quality in training up to 4k, i.e. the mean Δ over seeds 0–2 at 4k has upper 95% bound < 0.02.
- Otherwise run **R3b**: seeds 1 and 2 at T ∈ {1024, 4096} (dense, law, frac), to put error bars on the law vs fraction comparison.

---

# P3 cut theory, pre-registered 2026-10-30 (D-028) — `p3_cuts.py`, `p3_analyze.py`
**Theory** (`docs/cut_theory.tex`):
- **Output as a CGF derivative.** Each head's output is the h-derivative of the joint cumulant generating function of (score, value) under the uniform measure over keys.
- **Hot limit = mean-field expansion.** o = E v + κ(s,v) + ½κ(s,s,v) + …:
  - order 1 (`mf1`) is a centred linear-attention form, cost O(d²) per token;
  - order 2 is `mf2`.
- **Cold limit = saddle (top-k).** The error is exactly (1 − m_S)(⟨v⟩_dropped − ⟨v⟩_kept).
- **Hybrid = saddle + mean-field tail** (`hyb{k}`).
- **Prediction:** the score spread σ of each head picks the cheapest cut; only the crossover regime is expensive.

**Setup:**
- Frozen fp32 Pythia-160M, Pythia-410M and Qwen2.5-0.5B. T = 2048, WikiText validation, permutation seed 71.
- 8 calibration sequences: per-head tables, plus single-head and pair cuts on the first 4.
- 8 held-out sequences for evaluating the allocators.
- **Head-level error:** mean over queries of ‖Δo_h W_O^{(h)}‖², pushed through that head's slice of the output projection, under exact upstream activations.
- **Network effect:** ΔKL of the next-token distribution vs dense, mean over positions.

**Costs** (per query, d = head dim, n = row length):

| method | cost |
|---|---|
| full | 4nd |
| top-k | 4kd (+ 2nr selection, r = 16) |
| mf1 | 4d² |
| mf2 | 4d² + 2d²(d+1) |
| hyb | top-k + 8d² + 2kd |

**Primary cost model:** selection free. It is conservative, since it favours top-k. The indexer-costed model is secondary and reported descriptively; under it, top-k has a floor of r/2d = 12.5% of dense.

**Smoke run (T = 256, 2 sequences, Pythia-160M), recorded before the full run:**
- σ per head runs 1.25–76 (median 3.4);
- mf1 beats top-64 on no head;
- the next term predicts the mf1 error with rank correlation 0.92.

On synthetic Gaussian scores the expansion converges for σ ≲ 0.5 and diverges for σ ≳ 1. **So the hot regime may be nearly empty in trained LMs.** That is the main risk to P-A.

**Predictions:**
- **P-A (two regimes; σ decides):**
  - (i) mf1 has lower head-level error than top-64 (equal cost) on ≥ 5% of heads, in ≥ 2 of 3 models;
  - (ii) in models with ≥ 5 such heads, AUC(−σ → mf1 wins) ≥ 0.8.
  - Falsifier: (i) fails, meaning the hot side is practically empty in trained LMs at 2k.
- **P-B (error theory):**
  - (i) Spearman(next-term prediction, actual mf1 error) across heads ≥ 0.8, in every model;
  - (ii) Spearman(head-level error, single-head ΔKL) across heads × {topk8, topk32, mf1, hyb8} ≥ 0.6, in every model. This checks that local error predicts the global effect, which the allocator needs.
- **P-C (allocator):**
  - a greedy allocator over all methods, driven by predicted errors, gets eval KL ≤ 0.8 × the greedy top-k-only allocator (which uses exact errors);
  - at both B = 0.05 and 0.15 of dense attention FLOPs (selection free), in ≥ 2 of 3 models;
  - this isolates whether the mean-field and hybrid side buys anything.
  - Reported too: uniform top-k, a hand rule (layer 0 gets 8× k), and a measured-error allocator as reference.
- **P-E (faces, the spin-foam gluing):**
  - take the top-50 head pairs (layer a < layer b) by Elhage Q/K-composition score vs 50 random pairs, with both heads cut to top-8;
  - normalised interaction r = |ΔKL_both − ΔKL_a − ΔKL_b| / (ΔKL_a + ΔKL_b);
  - holds if median r_top ≥ 2 × median r_random and Mann–Whitney p < 0.01, in ≥ 2 of 3 models;
  - caveat: the composition score ignores RoPE and LayerNorm.

**Decision rules:**
- **P-A(i) fails:** trained LMs live in the cold/crossover regime and the mean-field cut is not free. That explains why linearising pretrained models needs retraining (LoLCATs-style). The theory stays as an explanation, and the focus moves to T2 theory (why γ < 1) and T1.
- **P-A and P-C hold:** this is the main contribution. Next comes a training test (a temperature-allocated hybrid model).
- **P-E holds:** spin-foam gluing earns its place, and the allocator gets a pair (face) correction.
- **P-E fails:** cut errors compose additively and faces don't matter for cutting.

**Amendment before the run (REV-007 + L4), 2026-10-30. Still pre-data: only the T = 256 smoke has been seen.**

Prior art (L4):
- **Scatterbrain (2110.15343)** already has temperature regimes for sparse vs low-rank, the exact-head + low-rank-tail hybrid, and an entropy-based per-layer budget split.
- **Linear attention as first-order Taylor** is in 2507.23632 and 2209.04881.
- **The top-k error identity** is in 2512.07647 (Thm 5.2).

So P-A is a *measurement* on trained LMs of a regime structure that is known in theory. What remains ours:
- the shift-invariant cumulant form and its σ convergence criterion (P-B(i));
- selecting the cut *type* per head from predicted errors (P-C);
- the face-interaction test (P-E).

The strongest literature baseline for P-C is a KL-guided layer selector (2512.20569). It is not implemented; our `measured_all` allocator, which uses measured head errors, is the stronger in-house reference.

Code fixes (Codex REV-007):
- Reference log-probs are kept on CPU; KL is computed in fp64 in 512-row chunks.
- Moments are computed in fp64.
- top-k masks are built from exact indices (no ties).
- `full` is allowed in every allocator, so top-k-only means {full, top-k}.
- Infeasible budgets are reported as infeasible.
- **P-E pairs are disjoint:** no head appears in two pairs, and the random pairs avoid the top heads. The pair floor ΔKL_a + ΔKL_b ≥ 1e-5 is pre-registered, with excluded counts reported. The test is scipy's Mann–Whitney (one-sided "greater").
- Final aggregate booleans implement the rules exactly: P-B in every model, P-C/P-E in ≥ 2 of 3, P-A(ii) over the eligible models.

**Amendment 2 (theory-note review), 2026-10-30. Still pre-data.**
- **Convergence.** The radius of the hot series is set by the complex (Lee–Yang) zeros of Z(τ) = Σ e^{τ s_j}. For Gaussian rows, R ≈ √(log n)/σ; at n = 2048 that is 2.76/σ. Longer rows enlarge the convergent region, so hot heads may exist at T = 2048 even though none did at T = 256. Recorded as a **descriptive prediction**: the fraction of heads with σ < √(log n) at their median row length.
- **Hybrid prediction:** now includes the first neglected log Z_tail term, dw·(ô_T − o_S) with dw = w(1−w)·κ₃,tail/6.
- **P-E redefined.** At leading order a KL interaction equals the linear-response term Cov_p(x_a, x_b), for any pair whose logit changes overlap. The test therefore uses the residual r = |ΔKL_ab − ΔKL_a − ΔKL_b − Cov_p(x_a, x_b)| / (ΔKL_a + ΔKL_b), with three disjoint groups of 20 pairs:
  - top Q/K-composed (faces);
  - top V-composed (chains, the control);
  - random.

  **Holds** if median r_top ≥ 2 × r_random, Mann–Whitney p < 0.01, and median r_top > median r_vcomp, in ≥ 2 of 3 models.

---

# T2-LONG — does the k*(T) law hold at 7–8B and up to 32k? Pre-registered 2026-10-30 (D-029) — `t2_long.py`
**Setup:**
- **Models:**
  - the Qwen2.5 size ladder: 0.5B, 1.5B, 3B, 7B;
  - Mistral-7B-v0.3;
  - Qwen3-8B.

  All frozen, with fp32 weights. Attention scores are always strict fp32. TF32 is allowed for the linear layers only if the precision gate passes.
- **Precision gate:** on Qwen2.5-7B, 2 books at 8k, mean KL(dense strict ‖ dense TF32) < 1e-4.
- **Data:** PG19 validation, 8 books, each starting 2000 tokens in. The lengths T ∈ {2k, 4k, 8k, 16k, 32k} are nested prefixes of the same segment. Long single documents avoid the stitched-article bias of WikiText at 32k.
- **Measures:**
  - oracle top-k with k ∈ {16, …, 4096};
  - k*(T) at ε ∈ {0.05, 0.02, 0.01} (mean next-token KL vs dense);
  - γ_F fitted over 2k–32k, with a 90% bootstrap CI over books.

**Hypotheses:**
- **H1 (generalises):** for every 7–8B model, γ_F(0.02) lies in [0.35, 0.70], i.e. its point estimate falls in the band. It **holds** if all three 7–8B models do.
- **H2 (no drift with size):** within the Qwen2.5 ladder, |γ_F(7B) − γ_F(0.5B)| < 0.15 at ε = 0.02. Spearman(γ, size) is reported.
- **H3 (a power law up to 32k):** the last local exponent (16k→32k) is within 0.2 of the global fit, for each 7–8B model at ε = 0.02.

**Decision rule:**
- H1 + H3 hold: T2 generalises, and the theory paper goes ahead.
- H1 fails: the result becomes "γ depends on scale/architecture", and the paper is rescoped.

**Theory data (exploratory, criteria fixed now):**
- σ²(n) is the per-query score variance and n_eff = 1/Σp², both as a function of row length n, averaged over heads, per layer, at T = 32k.
- **REM prediction:** if σ² ≈ a·log n + b, then n_eff ∝ n^{1−a}.
- **Checks:**
  - (i) a log-linear fit of σ²(n) over n ∈ [64, 32k];
  - (ii) |γ_neff − (1 − a)| ≤ 0.1, median over layers;
  - (iii) |γ_F − median-layer γ_neff| ≤ 0.1.

# T4-SELECT — can a cheap, GQA-shared, decode-only selector meet the k*(T) budget? Pre-registered 2026-10-30 (D-030) — `t4_select.py`
**Setup.**
- **Models:** Qwen2.5-7B, then Mistral-7B-v0.3, then Qwen3-8B. fp32 weights, TF32 matmuls outside attention, strict fp32 inside the tail attention.
- **Data:** PG19 validation, the same 8 books as T2-LONG; T ∈ {4k, 8k, 16k, 32k}.
- **Prefill:** the first T−256 tokens are dense (SDPA) and cached.
- **Decode-like tail:** the last 256 tokens are run against the cache. Only their attention is sparse, in every layer.
- **Metric:** KL of the next-token distribution vs the dense tail, averaged over the 256 rows.
- **Budget axis:** *tokens read per KV head per row*, measured as the union over the GQA group.

**Modes.**
- `tok`: per-query-head oracle top-k (calibration).
- `tok_gqa`: oracle on the group-max of log-probabilities, shared per KV head.
- `meanB`: block-mean key scores, B = 16 or 32.
- `questB`: Quest upper bound, Σ_d max(q_d·min_d, q_d·max_d), B = 16, 32 or 64.
- **Block modes always keep** the first max(16, B) tokens (sink) and the last 64 or more (window). Their block scores are normalised per head and max-pooled over the group.
- **Notation:** read*(ε) is the tokens read at KL = ε (log-log interpolation). γ is the log-log slope of read*(0.02) over 4k–32k.

**Needle (passkey) at T ∈ {8k, 32k}.**
- 4 books × depths {0.1, 0.5, 0.9}.
- Teacher-forced exact match of the 5-digit key.
- Run at that mode's own budget*(0.02) and at 2× that budget. Only trials the dense model gets right count.

**Selector choice.** "best" is the block mode with the lowest read*(0.02) at 16k on Qwen2.5-7B. It is then fixed for all models and for H2–H5.

**Hypotheses (per model, ε = 0.02, at T = 32k unless stated).**
- **H1 (shared selection):** read*(tok_gqa) ≤ read*(tok).
- **H2 (cheap selector is enough):** read*(best) ≤ 2 × read*(tok_gqa).
- **H3 (stays sub-linear):** γ(best) ≤ 0.85.
- **H4 (bandwidth):** read*(best)/T ≤ 10%.
- **H5 (retrieval):** at 2× budget, best keeps ≥ 11 of 12 dense-correct needles at each T. The 1× result is reported.

**Decision.**
- **H2–H5 hold on ≥ 2 of 3 models:** build the llama.cpp prototype. It adds a per-block key summary to the KV cache, then a block top-k, then `ggml_top_k_mask` (blocks + extra), then sparse flash attention, at decode only. Benchmark decode tok/s at 8k–64k with llama-bench on Metal and CUDA.
- **H2 fails but H1 holds:** the selector is the bottleneck. Fallback: exact K-scores, which save only V reads (~2×).
- **H5 fails:** the budget needs a safety factor or a retrieval-aware selector. Do not advertise k*(0.02) budgets for retrieval tasks.

## T4-SELECT amendment 1 (2026-10-30, before any T4 data; the user flagged retrieval as the key risk)
The single needle test (H5) is replaced by a **retrieval-budget measurement**.

**Tasks** (RULER-style, inserted into PG19 books):
- `single`: one magic number.
- `multikey`: the target plus 3 distractor names with their own numbers.
- `vartrack`: a 2-hop chain, VAR C = VAR B = VAR A = number, with the hops placed after the root.

**Design.**
- T ∈ {8k, 16k, 32k}; 4 books × 5 depths = 20 trials per task and T.
- The prompt is prefilled densely. The last prompt token and the answer tokens are sparse, which is conservative: in a real engine the first answer token comes from dense prefill.
- Success means a teacher-forced exact match of the answer tokens.
- Every mode except `tok` is swept over the whole budget grid.

**Definition.** k_ret(T, task, mode) is the smallest grid budget at which the success rate, among dense-correct trials, is ≥ 0.9, and it stays ≥ 0.9 at every larger budget.

**Fidelity tails.** Each fidelity cell also stores the 99th-percentile KL per row, the maximum KL per row, and the share of rows where the top-1 prediction flips.

**Hypotheses (replacing H5).** "best" is as before; "fidelity budget" means budget*(0.02) of the same mode and T.
- **H5a (fidelity budget is retrieval-safe up to 2×):** for best, k_ret ≤ 2× the fidelity budget for `single` and `multikey`, at every T, on ≥ 2 of 3 models.
- **H5b (theory, cold regime):** a single needle concentrates attention, so k_ret(single) grows more slowly with T than the fidelity budget. The slope over 8k–32k is lower for tok_gqa and for best, on ≥ 2 of 3 models.
- **H5c (selector confusion):** report k_ret(multikey)/k_ret(single) for tok_gqa vs best. The prediction is that block summaries are hurt more by distractors (ratio larger for best). This comes from the smoke test, so it is labelled post-hoc.
- **`vartrack`:** reported only. Base models often fail it even when dense.

**Decision change.** The llama.cpp budget rule becomes max(fidelity budget, k_ret(multikey)) × a safety factor. If H5a fails, the shipped default uses k_ret, not the fidelity budget.

## T4-SELECT amendment 2 / T2-LONG amendment (2026-10-30, budget: the GPU is available for about 3 more hours)
- **T2-LONG ladder deferred.** Qwen2.5-3B, 1.5B and 0.5B move to a later cheap 24 GB GPU session. H2 (size dependence) is pending, not failed.
- **T4 model order:** Qwen2.5-7B, then Mistral-7B, then Qwen3-8B (only if time allows).
- **T4 retrieval at T ∈ {8k, 32k} only.** H5b's slope therefore uses 2 points.
- **Model requirement:** the "≥ 2 of 3 models" rules become "both of the first 2 models" if Qwen3-8B does not finish.

## T4-SELECT amendment 3 (2026-10-30): selectors copied from the literature
**Why.** T4 fidelity on Qwen2.5-7B at 4–16k shows block summaries (mean/Quest) needing 2.3–3.0× the reads of the shared token oracle. The user asked whether the selector can be taken from the labs.

**Added training-free, token-level, GQA-shared modes** (scored with an approximate q·k, then top-k as in `tok_gqa`):
- `sparq16`, `sparq32`: exact keys on the r largest-|q| channels (SparQ, Ribar et al. 2023; Double-Sparsity-like).
- `kq4`, `kq2`: a 4- or 2-bit key copy, asymmetric, quantised per channel over 32-token groups (as in KIVI). This is the version used.
  - A per-token-group variant failed in the smoke test because of Qwen's outlier key channels. It is dropped and recorded here.

**Smoke test** (Qwen-0.5B, local, the basis for this amendment, so the hypotheses below are post-hoc):
- kq2 and kq4 are within 5% of the tok_gqa oracle at 1–4k;
- sparq32 within 10%;
- retrieval k_ret is equal to the oracle on all tasks.

**Run.** Qwen2.5-7B with modes {tok_gqa, sparq16, sparq32, kq4, kq2}, same setup as T4. It replaces the Qwen3-8B T4 slot, which would not finish within the GPU window.

**H2′.**
- At T = 32k, min(read*(kq2), read*(sparq32)) ≤ 1.25 × read*(tok_gqa).
- For that mode, k_ret(multikey) ≤ 2 × k_ret(tok_gqa) at 8k and 32k.

**Cost model to report.** Index read = bits/16 of K for kq*, and r/d of K for sparq*.

**Decision.** If H2′ holds, the llama.cpp prototype uses token-level selection with a 2-bit per-channel key shadow, in place of block summaries.
- **Order change (same day, before any lit-mode data):**
  - the literature-selector run on Qwen2.5-7B goes right after Qwen's T4;
  - the block-mode Mistral T4 is replaced by Mistral with {tok_gqa, mean16, sparq32, kq2}.
  - H2 (block selectors) is judged on Qwen2.5-7B alone.

---

## HOTCACHE — can a RAM hot cache serve a 2-bit selector's picks on a Mac? (pre-registered 2026-11-02, before any data; `src/hotcache_trace.py`, `src/hotcache_sim.py`, `src/ssd_bench.py`, local M4 Pro)
**Why.** L5 (R-LIT-L5) found that on unified memory the only slower tier is the SSD. Scattered per-token SSD fetches are far too slow, so the memory tier works only if the rows the selector picks are already in RAM almost every time. Nobody has measured this. It is stage B of the L5 plan: the go/no-go for an SSD KV tier.

**Setup.**
- Model: Qwen2.5-1.5B (28 layers, 2 KV heads, d = 128), fp32 on MPS. Data: PG19 validation, 4 books (as in T2-LONG).
- Final length T ∈ {8k, 16k, 32k}.
- The prefix of T − 1088 tokens is prefilled densely. Then a teacher-forced tail of 1088 rows: 64 warm-up rows followed by N = 1024 measured rows.
- Tail attention is sparse with `kq2` (2-bit per-channel K over 32-token groups, GQA-shared group-max log-prob, as in T4) at budget 1024.
- For every (layer, KV head, tail row), log the ranked top-2048 key indices of the kq2 selector, so that every budget k ≤ 2048 is a prefix. Also log the per-row KL against dense attention.

**Cache model (offline replay, `hotcache_sim.py`).**
- Always resident and never counted: sink (first 16 positions) and window (last 64 positions). New rows are written to RAM.
- Each (layer, KV head) has an LRU cache of capacity C (cache sizes are per layer).
- Items are tokens (g = 1) or 16-token blocks (g = 16, about 8 KB for a 7–8B layer/head pair at fp16, the SSD page scale).
- Within a step, higher-ranked picks count as more recent.
- The warm-up rows fill the cache; the hit rate is measured over the N measured rows only.
- Sweep: k ∈ {128, 256, 512, 1024, 2048}; C/T ∈ {2.5%, 5%, 10%, 20%}; g ∈ {1, 16}.
- Reported per cell:
  - row hit rate (selected rows already resident / selected rows);
  - compulsory-miss fraction (first-touch picks, which no policy can avoid);
  - misses per step;
  - spread across layers.

**SSD model (`ssd_bench.py`).** Random reads with `F_NOCACHE` from a 4 GiB file on this Mac's internal SSD, at 4 / 16 / 64 KiB and 1 / 8 / 32 threads, giving latency and throughput. The predicted fetch time per decoded token is (miss blocks per token over all layers and heads) / (measured IOPS at that block size), for the KV layouts of Qwen2.5-7B (28 × 4), Llama-3.1-8B (32 × 8) and Qwen2.5-14B (48 × 8). This assumes the 1.5B hit rates transfer, which is a caveat stated with the results.

**Primary cell.** T = 32k, k = 1024, g = 16, C = 10% of T.

**Decision rule** (on the 4-book mean hit rate in the primary cell):
- **GO** if ≥ 97%: stage C (the SSD tier) is worth building;
- **PARTIAL** if 90–97%: only with asynchronous prefetch or a speculative selector one step ahead (SpeCache-style); no direct build;
- **NO-GO** if < 90%: the SSD tier is dead for dense models on a Mac, and this is reported as a negative result.

**Hypotheses** (predictions written before any data):
- **HC1:** the primary cell lands in PARTIAL, between 90% and 97%.
- **HC2:** the hit rate rises with C and falls with k. At fixed C/T it does not fall by more than 3 points from T = 8k to 32k.
- **HC3:** blocks (g = 16) give a higher row hit rate than tokens at equal capacity in rows, because selected keys cluster.
- **HC4:** compulsory misses are ≤ 1% of picks in the primary cell, so any gap to 100% is a capacity or policy problem rather than a novelty problem.
- **HC5:** the per-layer spread is large. At least one layer is ≥ 10 points below the median layer, which would point to a mixed policy (pin some layers fully).

**Quality guard.** The mean tail KL at k = 1024 is reported. If it exceeds 0.05 at 32k, the trace is still valid for cache statistics, but the primary budget is flagged as too small for deployment.

---

## PREDICT-128K — does the k*(T) law predict 64k–128k, and does 2-bit selection hold up there? (pre-registered 2026-11-03, before any data; `src/t4_select.py`, H100 80 GB `vastai5`)
**Why.** The paper's main claim is k* ∝ T^γ. Kim (L6) assumes the budget is independent of T, so a fit over 2k–32k might not extrapolate. A law that predicts out of sample is what makes the claim ours. The llama.cpp prototype also needs evidence of quality at long context on the target layout (8 KV heads).

**Models.** Never measured by us, both natively ≥ 128k, both with 8 KV heads:
- Llama-3.1-8B (`unsloth/Meta-Llama-3.1-8B`, ungated mirror of Meta's base model);
- Mistral-Nemo-12B (`mistralai/Mistral-Nemo-Base-2407`).

Weights in bf16; attention math in strict fp32, as in T4.

**Setup (T4 decode-only).**
- PG19 validation books (test split if fewer than 4 are long enough), 4 books, OFFSET 2000.
- T ∈ {2k, 4k, 8k, 16k, 32k, 64k, 128k}.
- Dense prefill of T − 256 tokens, then a sparse teacher-forced tail of 256 rows.
- Modes: `tok` (per-head oracle), `tok_gqa`, `kq2`.
- Budget grid: 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384.
- Rows are processed in chunks of 64. This is row-wise identical to T4.

**Primary quantity.** k*(T) = `tok` budget*(0.02), the 4-book mean KL crossing.

**Prediction rule** (fixed now):
- Fit log k* = a + γ log T by least squares on T ∈ {2k, 4k, 8k, 16k}.
- Predict k* at 32k, 64k and 128k.
- The prediction interval comes from a book bootstrap of the fit (1000 resamples, 5–95%).

**Verdict per model:**
- **PASS** if the measured k* at both 64k and 128k is within ±25% of the prediction;
- **PARTIAL** if within ±50%;
- **FAIL** otherwise.

**The law is supported** if both models PASS or one PASSES and the other is PARTIAL.

**Saturation check (Kim's alternative).** The local exponent from 64k to 128k is reported. SATURATING if it is < 0.5 γ_fit on both models.

**Secondary** (also fit on 2k–16k, scored by the same rule):
- `kq2` read*(0.02).
- `tok_gqa` read*(0.02).
- The kq2/tok_gqa read ratio at 64k and 128k. H2′ from T4 is expected to hold: ratio ≤ 1.25.

**Retrieval.**
- Tasks: single, multikey, vartrack, as in T4 amendment 1.
- T ∈ {32k, 128k}; 4 books × 5 depths; modes `tok_gqa` and `kq2`; same budget grid.
- **Hypothesis PR1:** k_ret(kq2) ≤ 256 for single and multikey at 128k, on both models. That is, the retrieval budget stays flat out to 128k.

**Tails.** The 99th-percentile KL and the top-1 flip rate at budget*(0.02) of kq2, reported per T.
- **Hypothesis PR2:** the flip rate at kq2 budget*(0.02) does not grow by more than 3 points from 32k to 128k.

## PREDICT-128K amendment 1 (2026-11-03, before any experiment data; only smoke checks at 4k and 20k with ≤ 3 budgets on 1 book)
- **Precision: fp32 weights, not bf16.** The smoke check on Llama-3.1-8B at 4k (`tok`, 1 book) found that bf16 amplifies any perturbation into a KL floor.

  | budget | bf16 KL | fp32 KL |
  |---|---|---|
  | 3072 | 0.011 | 0.0004 |
  | 1024 | 0.035 | 0.024 |

  ε = 0.02 is too close to that floor. All earlier T2/T4 runs used fp32.
- **Model 2: Llama-3.2-3B (`unsloth/Llama-3.2-3B`) replaces Mistral-Nemo-12B.**
  - In fp32 at 128k, Nemo needs about 92 GB: 49 GB of weights plus 43 GB of KV.
  - Llama-3.2-3B is native 128k with 8 KV heads. It pairs with the 8B as a two-point size comparison within one family (the deferred T2 H2).
  - Nemo is dropped from this session.
- **Prefill is chunked** (8192 tokens, mem-efficient SDPA with an explicit causal mask). Check: KL between one-shot and chunked prefill tails is 3e-9 at 20k.
- **Unchanged:** the tail path (strict fp32 einsum), the prediction rule, the verdicts and the hypotheses.

## PREDICT-128K amendment 2 (2026-11-03, during the run, to save GPU time; the user approved skipping what no hypothesis needs)
**What had already been seen.** All fidelity data, the 32k retrieval for both models, and the 3B's 128k single-fact result (k_ret = 16).

**Changes.**
- **8B 128k retrieval out of memory.** It ran out of memory in the 8192-token prefill. It reran with prefill chunks of 4096 and `expandable_segments`. Its 32k retrieval per-trial data was lost: retrieval was saved only at the end. The k_ret summary lines survive in `logs/gpu5/pred_llama31_8b.log`. Retrieval is now saved after each task.
- **Variable-tracking dropped at 128k, both models.** No hypothesis uses it, and at 32k dense accuracy was only 11–12 of 20.
- **8B multikey at 128k uses a grid capped at 1024 keys.** The 8B takes about 1.9 h per 128k task. PR1 asks only whether k_ret ≤ 256, which a 1024 cap still decides; a k_ret beyond the cap is reported as "> 1024". The 8B single-fact task keeps the full grid.

## PORT-K0 (2026-11-03): can training-free sparse decode reach 2× on Metal? (pre-registered before any timing data)

**Question.** On the M4 Pro, how fast is the whole sparse attention path (score, select, build the mask, sparse FA with GQA bundling) compared with dense FA with GQA bundling? This is the fair baseline: upstream draft PR #30728.

**Setup.**
- llama.cpp branch `sparse-gqa`: `sparse-decode-design`, plus the 3 commits of PR #30728, plus one host change that allows GQA bundling on the sparse path for single-token decode (`ne01 == 1`).
- Correctness was checked against CPU before any timing: 6/6 OK at kv = 32k, n_kv_max 4176 / 8272 / 16384, f16 and q8_0.
- **Shape:** Llama-3.1-8B / Qwen3-8B attention (dk = dv = 128, 8 KV heads, G = 4), q8_0 K/V, one decode token. The sparse mask has randomly scattered indices.
- **Grid:** T ∈ {32768, 65536, 131072}; B = n_kv_max ∈ {2128, 4176, 8272, 16384}.
- `test-backend-ops perf`, env `SPARSE_K0=1`, 3 repeats, median µs per op. Runs under `run_scripts/guarded_cmd.sh` (kill if free memory < 20% or swap grows by > 1.5 GB).

**Components, per layer, all measured with existing ops:**
- `a_dense(T)`: FA with n_kv_max = 0, with `GGML_METAL_FA_VEC_GQA=1`. Stock (GQA = 0) is also reported.
- `a_sparse(T, B)`: FA with n_kv_max = B and GQA = 1.
- `s(T)`: the scorer, a 4-bit stand-in: `mul_mat` q4_0, T × 128 K per KV head against the G = 4 query columns. It is an upper bound for the planned 2-bit scorer.
- `sel(T, B)`: `top_k` f32 over T per KV head (8 rows), k = B − 80, rounded down to the grid {2048, 4096, 8192, 16384}.
- `m(T, B)`: `top_k_mask` building per-KV-head masks from B cell indices.

**Projection (end-to-end decode per token, Llama-3.1-8B Q4_K_M, L = 32).**
- `t_dense = W + L·a_dense`.
- `t_sparse = W + L·(a_sparse + s + sel + m)`.
- `W = 4.92 GB / 176 GB/s = 27.8 ms`, using the effective weight bandwidth from R-HEADROOM.
- Other per-token costs are ignored on both sides.
- `S(T, B) = t_dense / t_sparse`.

**Primary cell:** (T = 131072, B = 8272). The measured 8B r_KV* at 128k is about 7.7k plus sink and window. Secondary cells: (32768, 2128) and (65536, 4176).

**Verdict.**
- **PASS:** S(128k) ≥ 2.0 and S(32k) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0. Proceed only if a bandwidth bound for a 2-bit scorer plus threshold selection closes the gap; that bound is reported as [expl].
- **KILL:** S(128k) < 1.5.

**Also reported:**
- `a_dense / a_sparse` (kernel only);
- the sparse speedup from GQA bundling (`a_sparse` at GQA 0 vs 1);
- the component breakdown.

**Caveats:**
- The test mask has one row broadcast over heads, so the index-gather pass is 1/8 of a per-KV-head setup. It is small next to KV reads and is reported as is.
- The random index pattern is a worst case for locality.
- No quality is measured here; quality comes from R-PREDICT-128K (kq2 within 10% of the oracle r_KV*).

## PORT-K1 (2026-11-03): page-level sparse decode on Metal (pre-registered before any K1 timing data; K0 results seen)
**Motivation.** R-PORT-K0 failed on two costs:
- the token-level scorer plus top-k, which cap S at 1.84;
- the scattered gather, at 4× dense cost per key.

Page-level selection (MOIRA 2610.04313) reads contiguous 64-token pages and scores per-page statistics instead of keys. This test asks whether that reaches 2× on the M4 Pro. **Speed only;** quality is MOIRA's published claim and is not tested here.

**Setup.**
- llama.cpp branch `sparse-gqa`, with the sparse n_kv_max cap raised to 32768. Correctness was checked against CPU at 24576 before timing.
- Shape and KV type as in K0: Llama-8B, q8_0, one decode token, GQA bundling on.
- `test-backend-ops perf`, env `SPARSE_K1=1 SPARSE_BLOCK=64`, 3 repeats, median, under `guarded_cmd.sh`. `SPARSE_BLOCK=64` makes the test mask select B/64 random whole 64-cell pages.
- **Grid:** T ∈ {32768, 65536, 131072}; B ∈ {4096, 8192, 12288, 16384, 20480, 24576, 32768}, with B < T.

**Components per layer.**
- `a_dense(T)`: FA, GQA = 1.
- `a_sparse(T, B)`: FA over B cells in whole pages, GQA = 1.
- `s(T)`: page scorer stand-in. `mul_mat` f16 of T/64 page rows × 256 (per-channel mean and variance) per KV head, against G = 4 query columns (q and q², concatenated).
- `sel(T, B)`: `top_k` f32 over T/64 pages per KV head, k = B/64.
- `m(T, B)`: `top_k_mask` with 64-cell blocks, real cache layout.
- Page-statistics upkeep (once per 64 tokens) is ignored.

**Projection:** as in K0. W = 27.8 ms, L = 32, `S = (W + L·a_dense) / (W + L·(a_sparse + s + sel + m))`.

**Budgets, from MOIRA's reported kept fraction.** MOIRA reports 42% at 8k and 15% at 130k; we interpolate log-linearly in T.
- Primary: (131072, 20480), 15.6%.
- Secondary: (32768, 8192), 25%; (65536, 12288), 19%.

**Verdict.**
- **PASS:** S(128k, 20480) ≥ 2.0 and S(32k, 8192) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0.
- **KILL:** S(128k) < 1.5.

**Also reported:**
- the break-even B per T (S = 1);
- the largest B with S ≥ 2 at 128k;
- the per-key cost of the page gather against the token gather (K0) and dense streaming.

**If PASS, next:** an offline quality check of the page mean/variance selector at matched kept fraction, on our PG19 harness (KL at 32k–128k, retrieval), before any integration.

## PORT-K1 amendment 1 (2026-11-03, before any K1 timing data; only the correctness gate had run)
- **The f16 sparse FA at n_kv_max = 24576 misses the harness NMSE threshold:** 5.2–5.5e-4 against 5e-4.
  - The miss is the same with GQA bundling off or on, and with token or 64-cell page masks. It is f16 accumulation error over about 24k keys in the existing sparse kernel, not our change.
  - All q8_0 cases pass, including 24576.
- K1 times q8_0 only, so the gate now requires the 4 q8_0 eval cases to pass.

## PORT-K2 (2026-11-03): a fast sparse FA kernel (pre-registered before any K2 timing data; K0 and K1 results seen)
**Motivation.** In K1 the sparse FA cost 65–85 µs per 1k keys against about 20 for dense, independent of access pattern.
- Code inspection: the sparse path uses `fa_vec_baseline_cfg` (NE = 1 for dk = 128: one KV row per simdgroup step).
- Dense decode on Apple9 (M4 Pro) uses the tuned config (Q = 1, NE = 4: four rows in flight per simdgroup).
- The inner loops are otherwise the same except for the row index.

**Variants.**
- **V1 (primary):** the sparse path uses the tuned config with Q forced to 1. It is env-gated, `GGML_METAL_FA_SPARSE_TUNED=1`, and the bucket is n_kv_max.
- **V2 (built and measured only if V1 is not PASS):** a page-aware sparse kernel that reads one page start per 32-cell chunk and uses dense pointer arithmetic inside the page.

**Setup.**
- As in K1: page masks (`SPARSE_BLOCK=64`), the same grid, components and projection, `port_k1_analyze.py` logic, and results in `results/port_k2/`.
- Correctness against CPU is checked before timing: the q8_0 cases at 4176 / 8272 / 16384 / 24576, with token and page masks.
- Secondary: the K0 token-level grid rerun with V1, labelled [secondary].

**Verdict (V1, then V2 if run):**
- **PASS:** S(128k, 20480) ≥ 2.0 and S(32k, 8192) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0.
- **KILL:** S(128k) < 1.5.

**Also reported:** µs per 1k keys of sparse vs dense.

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

## PORT-K3 (2026-11-03): page-level index gather (pre-registered before any K3 timing data; K2 results seen)
**Motivation.** K2 V2 leaves a fixed cost that grows with T (about 40 µs at 32k, about 115 µs at 128k). One source is the index-gather pass, which scans all T mask cells in one threadgroup per row.

**V3 = V2 + `kernel_flash_attn_ext_vec_idx_pages`.**
- It reads only the first cell of each 32-cell chunk and writes the chunk start at list slot 32·rank.
- Selected with `GGML_METAL_FA_SPARSE_PAGED=1`.
- Correct against CPU for q8_0 at all four budgets; f16 is OK except the known 24576 rounding case.

**Setup, components, projection and verdict rules:** identical to K1 and K2. Page masks, results in `results/port_k3/`.
- **PASS:** S(128k, 20480) ≥ 2.0 and S(32k, 8192) ≥ 1.4.
- **PARTIAL:** 1.5 ≤ S(128k) < 2.0.
- **KILL:** S(128k) < 1.5.

## PORT-E1 (2026-11-04): in-engine page selection in llama.cpp is correct (pre-registered before any in-engine data; PAGE-Q and K3 results seen)

Question: does page selection inside the llama.cpp graph (Metal, page-gather FA kernel from K3) select the pages PAGE-Q selects and give the KL PAGE-Q measured? This is a correctness gate before any tokens/s claim. Stage A only: page statistics are recomputed from the whole K cache each decode step (no speedup expected); the incremental page cache is PORT-E2.

Engine: branch `sparse-gqa` of the local llama.cpp clone. Env `LLAMA_SPARSE_PAGES=<budget B>`, `LLAMA_SPARSE_SCORE=mv|quest|mean`, decode steps with one token only. Same rule as `t4_select.select_mask` block mode (b = 64): eligible pages 1 ≤ p < win_blk with win_blk = max(0, ⌊(pos − 63)/64⌋); per-query-head page score (mv: s·μ + ½ s²·σ², quest: s⁺·max + s⁻·min, mean: s·μ, with s = q·kq_scale); per-head log-softmax over eligible pages; max over the G heads of the group; top nb = ⌊(B − 80)/64⌋ pages per KV head; keep page 0, the picked pages, and cells win_blk·64 … pos. Metal env `GGML_METAL_FA_SPARSE_TUNED=1 GGML_METAL_FA_VEC_GQA=1 GGML_METAL_FA_SPARSE_PAGED=1`.

Model and data: unsloth/Llama-3.2-3B converted with `convert_hf_to_gguf.py` to f16 GGUF, KV cache f16, flash attention on. The 4 PAGE-Q fidelity books (same `book_segments` call, length 131072, validation split), truncated to T. Dense prefill of T − 256 tokens, then 256 teacher-forced single-token decode steps, as in P-T4. KL(dense ‖ sparse) per row in float64 over the full vocabulary, both runs in-engine.

Gates (all must pass; any failure is reported and fixed before E2):
- E1a exact selection, T = 8192, book 0, layers {0, 9, 18, 27}, decode steps {0, 255}, B ∈ {1024, 4096}, mv and quest. Python recomputes page statistics from the dumped f16 K (cast to f32) and the picks from the dumped q in float64. PASS if statistics max abs error ≤ 1e-3 and the picked page set equals the reference set, except for pages whose reference group score is within 1e-3 of the nb-th score.
- E1b machinery floor, T = 8192, 4 books: with B ≥ T (every eligible page picked) mean KL ≤ 1e-4; page kernel (PAGED=1) vs token kernel (PAGED=0) at B = 2048, mean KL between them ≤ 1e-5.
- E1c fidelity vs PAGE-Q, T = 32768, 4 books, mv64, B ∈ {1024, 2048, 4096, 8192}: per book and budget the ratio of engine KL to PAGE-Q KL (`pageq_llama32_3b` summary). PASS if the median of the 16 ratios is in [0.8, 1.25] and every ratio is in [0.5, 2.0]. Reported, no gate: quest64 at the same cells, and engine r* by the PAGE-Q crossing rule.

Memory: everything under `guarded_cmd.sh`. If T = 32768 with f16 weights is killed, rerun E1c with q8_0 weights (KV still f16) and say so; if that is also killed, E1c waits for a bigger machine.

## PORT-E1 amendment 1 (2026-11-04, before any E1 data on the protocol model; development runs on Qwen2.5-1.5B q8_0 seen)

Seen: on Qwen2.5-1.5B (not the protocol model) the E1a check found 0 differing pages in 32/32 cases and 0 differing mask cells in 8/8; page kernel vs token kernel mean KL 1.6–8.8e-6 per book (4 books, B = 2048, T = 8192).

Change: E1a–E1c use q8_0 weights (KV cache stays f16). Loading the f16 3B GGUF was killed twice by the memory guard (free 36% → 20% and 53% → 18%; the Mac already had ~20 GB of swap in use). If memory is freed before E1c runs, E1c is run with both f16 and q8_0 weights and both are reported; the gate applies to f16.

## PORT-E2 (2026-11-04): cached page statistics are exact (pre-registered before any E2 data on the protocol model; development runs on Qwen2.5-1.5B seen)

Engine: `LLAMA_SPARSE_PAGE_STATS=f32|f16` keeps [mean, var, max, min] of K per 64-cell page in the KV cache. A page is refreshed when a ubatch completes it and at every decode step in which it is the newest eligible page. `LLAMA_SPARSE_STATS=full` is the stage-A reference (recompute from the whole cache).

Seen on Qwen2.5-1.5B q8_0, T = 8192, 1 book: f32 cache bit-identical logits to full for mv and quest; f16 cache bit-identical for quest, mv mean KL 2.3e-5 (max 1.4e-3).

Gates on Llama-3.2-3B (q8_0 weights, KV f16), T = 8192, 4 books, configs mv:2048, quest:2048, mv:4096, quest:4096:
- E2a f32 cache: logits bit-identical to full, every config and book.
- E2b f16 cache: quest bit-identical; mv mean KL vs full ≤ 1e-4 per config (mean over 4 books).

## PORT-E1 amendment 2 (2026-11-04, before any E1c data; E1a, E1b and E2 results on Llama-3.2-3B at 8k seen)

E1c uses the f32 page cache (`LLAMA_SPARSE_PAGE_STATS=f32`) instead of full recompute. E2a showed it gives logits bit-identical to full recompute on 4 configs × 4 books, and it is about 8× faster per tail. Gates, model, books and budgets are unchanged.

## PORT-C (2026-11-05): page-sparse decode on CUDA (pre-registered before any CUDA data; all Metal PORT-K/E results, the 1.5B llama-bench smoke run and a Metal run of the new test cases seen)

Question: does the page-sparse decode built for Metal give the same selection, the same KL and a tokens/s gain on a consumer NVIDIA GPU, and does a cost model calibrated at short context predict that gain at long context?

Engine: llama.cpp branch `sparse-cuda` (4b281fd64, private backup). The in-graph selection (PORT-E1/E2) is unchanged. CUDA changes:
- The vector FA kernel accepts one mask per q head (`mask->ne[2] > 1`), decode only. Upstream CUDA rejects such masks.
- `GGML_CUDA_FA_SPARSE_PAGED=1`: a prepass builds, per q head, the list of visible 32-cell chunks (a chunk is visible iff its first cell is); the vector kernel then gives each warp one chunk from the list. This is the CUDA analogue of Metal K3.

Hardware: one RTX 4090 24 GB (Ada, sm_89), CUDA 12.x, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`. If no 4090 is available, another Ada card, stated in the results; the numeric bands below are for the 4090. One GPU job at a time.

Dense baseline everywhere: the same binary without the sparse env, so the stock kernel choice (the vector kernel for these one-token decode shapes on Ada).

**C1 correctness (gate before C2–C4).** `test-backend-ops -o FLASH_ATTN_EXT -b CUDA0` with `SPARSE_CUDA_EVAL=1 SPARSE_MASK_HEADS=1`: 16 cases (gqa 3 and 4, kv 8192 and 32768, n_kv_max 1056 and 4096, f16 and q8_0 KV), each under `SPARSE_BLOCK` ∈ {32, 64} × `GGML_CUDA_FA_SPARSE_PAGED` ∈ {0, 1}. PASS if all 64 pass the harness tolerance against CPU. Negative control: PAGED=1 with token-level masks (`SPARSE_BLOCK` unset, not chunk-aligned) must fail at least 8 of 16, showing the paged path is active.

**C2 kernel speed.** `test-backend-ops perf` with `SPARSE_CUDA_PERF=1`: dense runs plain; sparse runs with `SPARSE_MASK_HEADS=1 SPARSE_BLOCK=64 GGML_CUDA_FA_SPARSE_PAGED=1`. kv ∈ {8k, 16k, 32k, 64k, 128k}, B ∈ {2048, 3072, 4096, 6144, 8192, 12288, 14336, 16384, 24576, 27648, 28672, 32768} (B < kv), gqa 3 and 4, f16 and q8_0. Median of 3 invocations.
- H-C2a: the sparse per-key cost (slope of time over B at kv = 128k) is within 1.5× of the dense per-key cost (slope over kv), in each of the 4 (gqa × KV type) cells.
- H-C2b: S_attn(128k, 16384) = t_dense / t_sparse ∈ [4, 8] in each cell.
- PASS both hypotheses in all 4 cells; PARTIAL in 2–3 cells; KILL in fewer than 2.

**C3 end-to-end tokens/s.** `llama-bench -fa 1 -p 0 -n 64 -r 5 -d T`, decode tokens/s, dense vs sparse (`LLAMA_SPARSE_PAGES=B LLAMA_SPARSE_SCORE=mv LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`).
- Models: Llama-3.2-3B q8_0, KV f16 (the PORT-E model); Llama-3.1-8B q8_0 (`unsloth/Meta-Llama-3.1-8B`, the PAGE-Q checkpoint, converted the same way), KV q8_0 (f16 KV does not fit 24 GB at 128k).
- B per T: 2048 at 8k (the E1 budget); otherwise the PAGE-Q mv64 r*(T) at KL 0.02 rounded up to a multiple of 1024. 3B: 4096 / 14336 / 28672 at 32k / 64k / 128k. 8B: 6144 / 12288 / 27648.
- Calibration depths: 8k and 16k (3B B = 2048 and 3072; 8B B = 2048 and 3072). Test depths: 32k, 64k, 128k.
- Cost model (H-C3a, the main prediction): t_sparse(T, B) = t_dense(T) − L·[a_dense(T) − a_sparse(T, B)] + o0 + o1·T. Here a_dense, a_sparse are the C2 per-call FA times at exactly these (T, B) points (same box; gqa 3 and f16 KV for the 3B, gqa 4 and q8_0 KV for the 8B), L the layer count (28 and 32), and o0, o1 the selection overhead per token fitted exactly from the two calibration depths. PASS if the predicted sparse tokens/s is within ±10% of the measured value at every test depth, for both models. Fitted o0, o1 are reported.
- H-C3b (bands, from bandwidth arithmetic before any CUDA data): S = sparse / dense tokens/s.
  - 3B: 8k [0.8, 1.1]; 32k [1.1, 1.6]; 64k [1.3, 2.0]; 128k [1.7, 2.5].
  - 8B: 32k [0.9, 1.2]; 64k [1.0, 1.4]; 128k [1.2, 1.7].
- Usefulness verdict: PASS if S(3B, 128k) ≥ 1.5 and S(8B, 128k) ≥ 1.2; PARTIAL if only one holds; KILL if neither.

**C4 fidelity on CUDA** (`llama-sparse-tail`, as PORT-E1; f32 page cache, PAGED=1).
- C4a backend equivalence: 3B, T = 8192, 4 books, mv:2048, quest:2048, mv:4096, quest:4096. PASS if each config's mean KL (over 4 books) is within ±5% relative of the Metal value in `results/port_e1/e2_full_*.jsonl` (mv:2048 0.0168, quest:2048 0.0165, mv:4096 0.0061, quest:4096 0.0041).
- C4b = PORT-E1 gate E1c run on CUDA instead of Metal (backend change only; gates, model, books, budgets, f32 cache unchanged).
- C4c 128k fidelity, 3B, KV f16, 4 books, mv64 B ∈ {8192, 16384, 32768}: per book and budget the ratio of engine KL to PAGE-Q KL at 131072. Same rule as E1c: median ratio in [0.8, 1.25] and every ratio in [0.5, 2.0]. Reported, no gate: quest64 at the same budgets and engine r* by the PAGE-Q crossing rule.
- Not in this protocol: 8B fidelity at 128k with KV f16 (needs ≥ 32 GB of VRAM).

Order: C1, then C4a, C2, C3, C4b, C4c. A failed C1 stops everything until it is fixed; the fix and the failure are reported.

## PORT-C amendment 1 (2026-11-05, before any CUDA data; a Codex static review of the CUDA diff seen)

Engine is `sparse-cuda` 66ed860e7. It fixes three review findings outside the paged path:
- the dense mask scan no longer runs on per-head masks (it bounded every head by head 0's last visible key when there are several sequences);
- per-head masks with logit softcap go to the vector kernel only for D = 128 or 256;
- paged mode is compiled for CUDA only, not HIP/MUSA.

C1 gains one case, two sequences with per-head masks (gqa 3, kv 8192, n_kv_max 1056, f16): 17 cases, 68 runs, all must pass. The negative control still needs at least 8 failures. Everything else is unchanged.

## PORT-C amendment 2 (2026-11-05, after C1 and before any C2–C4 data)

The Vast box (`vastai7`) has an RTX 4090 with 48 GB, a memory-modified AD102 board (sm_89; driver 595.104, CUDA 13.2), not the 24 GB card. Chip and nominal bandwidth are those of the 4090, so the C2 and C3 bands still apply as written. Models, KV types and budgets are unchanged (the 8B keeps KV q8_0), so nothing depends on the extra memory. The 3B q8_0 file rebuilt on the box has the same SHA-256 as the local PORT-E file.

C1 result (seen before this amendment): PASS. 17/17 cases in each of the 4 runs, and the negative control failed 17/17.

## PORT-C2 (2026-11-05): 8B fidelity and retrieval at 128k inside llama.cpp on CUDA (pre-registered before any C2 data; PORT-C C1, C4a and 2 of 3 C2 timing runs seen)

Question: does the in-engine page rule reproduce PAGE-Q's 128k fidelity on the primary model (Llama-3.1-8B), and does in-engine retrieval at 128k need the same budget PAGE-Q found? Both are numbers PAGE-Q's GO decision rests on; so far only the Python reference has produced them.

Engine and box: `sparse-cuda` 310d7c858. It adds a retrieval mode to `llama-sparse-tail` (`--ret`); the CUDA kernels are unchanged from PORT-C. Box `vastai7` (48 GB RTX 4090). Weights q8_0 (the 8B file from PORT-C setup), KV f16, f32 page cache, `GGML_CUDA_FA_SPARSE_PAGED=1`.

**C5a 8B fidelity at 128k.** `llama-sparse-tail`, T = 131072, the 4 PAGE-Q books (`books128k.i32`, same tokens for both models), configs dense, mv:{8192, 16384, 32768}, quest:{8192, 16384, 32768}. Per book and budget, the ratio of engine KL to PAGE-Q KL (`pageq_llama31_8b`, 131072). PASS if the median of the 12 mv ratios is in [0.8, 1.25] and every mv ratio is in [0.5, 2.0] (the E1c rule). Reported, no gate: quest ratios, engine r* by the PAGE-Q crossing rule. Note: C4a found CUDA KL about 4e-4 above Metal at 8k. That is under 3% of the PAGE-Q 8B KL at B = 32768 (0.015), so well inside the gate.

**C5b retrieval at 128k, 8B and 3B.** The 40 PAGE-Q trials (single and multikey; 4 books × 5 depths each), rebuilt by `e2_ret_tokens.py` with PAGE-Q's construction code and RNG seed. Dense prefill of the prompt, then the answer tokens teacher-forced one decode step at a time. A trial is correct if every answer token is the argmax.
- Configs: dense, and mv and quest at {4096, 8192, 12288, 16384, 24576, 32768}, plus mv at the PORT-C C3 128k budget (27648 for the 8B, 28672 for the 3B).
- k_ret as in PAGE-Q: the smallest grid budget from which every larger grid budget has rate ≥ 0.9 over dense-correct trials. Grid: {4096, 8192, 12288, 16384, 24576, 32768}.
- Gates, per model and task:
  - (i) dense-correct trials ≥ PAGE-Q's minus 2. 8B: single ≥ 18, multikey ≥ 17. 3B: single ≥ 18, multikey ≥ 16.
  - (ii) engine mv k_ret within one grid step of PAGE-Q's mv64 k_ret. 8B: 12288 single and multikey. 3B: 12288 single, 8192 multikey.
  - (iii) at the C3 128k budget, mv rate ≥ 0.9 over dense-correct trials.
- PASS if (i)–(iii) hold for both tasks and both models. Each failing cell is reported.

Order on the box: after the PORT-C stages and the C4a diagnostic. C5a first, then C5b for the 8B, then the 3B.

## PORT-C3 (2026-11-05): last CUDA engineering round, two selection kernels fixed (pre-registered before any timing of the fixed build; R-PORT-C, R-PORT-C2 and the exploratory stage timings X-PORT-C-DRY seen)

Seen. Stage costs on `vastai8` (48 GB RTX 4090), 3B, KV f16, in ms per token over dense at 8k / 32k / 128k:
- stats update: 0.86 / 0.96 / 1.03
- scoring: 0.10 / 0.59 / 1.01
- log-softmax and group max: 0.72 / 0.97 / 2.78
- top-k: 0.12 / 0.39 / 1.35
- repeat and mask: 1.02 / 1.42 / 2.26

Full sparse: 7.85 / 9.36 / 17.29 ms, against dense 5.75 / 8.61 / 20.72.

Change (llama.cpp `sparse-cuda` d96e8140f), in CUDA kernels only; the graph and the selection rule are unchanged:
- `pool_1d` with one window per row (OW = 1) uses a block reduction per row. Before, one thread scanned the whole row: 2048 pages at 128k.
- `top_k_mask` runs as a fill kernel with many blocks per row, then a scatter kernel in which 64 threads write each picked page's cells. Before, there was one block per row: 24 blocks at decode.

Gates, all on `vastai8`:
- **K-gate (correctness):** test-backend-ops `-o POOL_1D` and `-o TOP_K_MASK` on CUDA0 against CPU all pass. PORT-C C1 (sparse FA eval, 4 runs) still passes.
- **F-gate (fidelity unchanged):** `llama-sparse-tail`, 3B, T = 8192, 4 books, mv:2048 and quest:2048, f32 page cache. Mean KL within ±1% relative of the PORT-C C4a values (0.017462, 0.017085).
- **S-gate (speed, the decision):** `llama-bench` as in PORT-C C3, 3B, KV f16, depths 8k / 32k / 128k, budgets 2048 / 4096 / 28672, dense and sparse in the same run.
  - Prediction: S(128k) = 1.6, band [1.45, 1.75]; S(32k) in [0.95, 1.2]; S(8k) in [0.75, 0.95].
  - **Engineering success if S(3B, 128k) ≥ 1.5.** Otherwise the engineering phase ends with the current result.

There are no further rounds after this one, whatever the outcome. The next steps are the write-up and, if wanted, the Metal test.

## PORT-C3 amendment 1 (2026-11-05): the 8B on the fixed build (written before any 8B timing on it; all PORT-C3 3B results seen)

Same build (d96e8140f), box `vastai8`, `llama-bench` exactly as PORT-C C3 for the 8B: Llama-3.1-8B q8_0, KV q8_0, depths 8k / 32k / 128k, budgets 2048 / 6144 / 27648, dense and sparse in the same run. The 8B q8_0 file is rebuilt on this box. Its SHA-256 is compared with the PORT-C file from `vastai7` (38a8de72…), and a mismatch is reported.

Prediction, from the old C3 8B times and the 3B savings scaled by 32 layers and 32 heads: S(8B, 128k) = 1.30, band [1.15, 1.45]; S(32k) in [0.9, 1.1]; S(8k) in [0.75, 0.95]. Usefulness bar, unchanged from PORT-C: S(8B, 128k) ≥ 1.2. This adds measurements only; the engineering phase stays closed.

## PORT-G (2026-11-05): a second GPU (RTX 3090) and a second model family (Qwen3-8B), with speedups predicted from short-context calibration (pre-registered before any 3090 or Qwen3 data; all PORT-C, C2 and C3 results and a retrospective check of the procedure on the 4090 seen)

Question: does the PORT-C3 result carry over to another consumer GPU and another model family? Is the long-context speedup predictable from measurements at ≤ 32k on the target GPU?

**Engine and hardware.**
- llama.cpp `sparse-cuda` 86c006aca. That is d96e8140f plus a `--kv-type` option in `llama-sparse-tail`; no kernel change.
- One RTX 3090 24 GB (Ampere, sm_86), on Vast. The exact board is stated in the results.

**Models (`setup_port_g.sh`).**
- Llama-3.2-3B and Llama-3.1-8B q8_0. Their SHA-256 must equal the PORT-C files.
- Qwen3-8B-Base, converted with Qwen's YaRN setting (`rope_scaling` yarn, factor 4, original 32768; `max_position_embeddings` 131072), then q8_0. Its SHA-256 is recorded.

**Retrospective check, seen (4090, exploratory).** Calibrating the cost model on 8k and 32k predicts S(128k) = 1.318 for the 8B (measured 1.313) but 1.25 for the 3B (measured 1.55). The 3B's 32k sparse point was noisy (±6.5%). Hence three calibration depths, 10 repetitions, and a least-squares fit.

**G1, predicted speedups (`run_port_g.sh`, `src/port_g_predict.py`).**
- Phase 1 on the 3090:
  - FA microbenchmarks: dense, plus group-shared page masks (`SPARSE_MASK_GROUP`, gqa 3 and 4, f16 and q8_0);
  - `llama-bench` dense decode tokens/s at 8k / 16k / 32k / 64k / 128k (r = 10 up to 32k, r = 5 above);
  - sparse at 8k / 16k / 32k only (r = 10).
- Models and KV types: 3B with KV f16; 8B and Qwen3 with KV q8_0.
- Budgets: 3B 2048 / 3072 / 4096 / 14336 / 28672; 8B and Qwen3 2048 / 3072 / 6144 / 12288 / 27648.
- Layers: 3B 28, 8B 32, Qwen3 36. gqa: 3, 4 and 4.
- `port_g_predict.py` gives the predicted sparse tokens/s at 64k and 128k:
  - t_sparse(T) = t_dense(T) − L·[a_dense(T) − a_sparse(T, B)] + o0 + o1·T;
  - o0 and o1 are fitted by least squares on the three calibration depths;
  - a are the microbenchmark times.
- The predictions are committed before phase 2 runs.
- Phase 2: sparse at 64k and 128k (r = 5).
- **H-G1: |pred_S / meas_S − 1| ≤ 10%** at 64k and 128k for each of the 3 models (6 points). PASS if all 6 hold; PARTIAL if 4–5; FAIL if fewer than 4.
- Reported against the PORT-C bars, not gated: S(128k) for the 3B (≥ 1.5), the 8B (≥ 1.2) and Qwen3 (≥ 1.2).

**Q1, Qwen3 fidelity at 128k (in-engine; there is no Python reference for Qwen3).**
- `llama-sparse-tail`, T = 131072, KV q8_0, f32 page cache, the same 4-book `book_segments` call tokenised with Qwen's tokenizer (book identity may differ from Llama's).
- Configs: dense, mv:{8192, 16384, 27648, 32768}, quest:27648.
- **Gate: mean KL at mv:27648 ≤ 0.03.** That is the Llama budget rule carried over unchanged.
- Reported: the KL curve and r*(0.02) by the PAGE-Q crossing rule.

**Q2, Qwen3 retrieval at 128k.**
- The 40 PAGE-Q trials (single and multikey), rebuilt by `e2_ret_tokens.py` with Qwen's tokenizer.
- Configs: dense, mv and quest at {4096 … 32768}, and mv:27648.
- **Gate: for each task with ≥ 10 dense-correct trials, the mv rate at 27648 is ≥ 0.9 over dense-correct trials.** A task with fewer than 10 is reported as uninformative.

Order: setup, phase 1, predictions committed, phase 2, analysis. RULER is not part of this protocol.

## PORT-R (2026-11-05): RULER at the operating point, a speed–budget curve, and wall-clock with prefill, on an RTX 4090 (pre-registered before any PORT-R data; PORT-C, C2, C3 seen, PORT-G running in parallel on a 3090)

**Box and engine.**
- `vastai11`: RTX 4090 24 GB.
- llama.cpp `sparse-cuda` 8213b023d. Kernels as in C3 (d96e8140f); `llama-sparse-tail` adds `--kv-type` and `--gen`, a greedy generation mode.
  - `--gen` prefills the prompt minus its last token densely.
  - Then, for each config, it decodes the last prompt token and up to `max_new` greedy tokens. Generation stops early at EOS.
- Models from `setup_port_g.sh`. The SHA-256 of each file is recorded, and a mismatch with the PORT-C files is reported.

**R1, RULER at 128k, Llama-3.1-8B q8_0, KV q8_0.**
- Data: NVIDIA RULER synthetic, all 13 tasks (niah_single_1–3, niah_multikey_1–3, niah_multivalue, niah_multiquery, vt, cwe, fwe, qa_1, qa_2).
  - 20 samples per task, seed 42, `max_seq_length` 131072 including generation, `base` template, tokenizer `unsloth/Meta-Llama-3.1-8B`.
  - RULER commit recorded by `ruler_prep.sh`.
  - The prompt is `input + answer_prefix`, tokenised with BOS (`port_r_tokens.py`). `max_new` follows RULER: niah 128, vt 30, cwe 120, fwe 50, qa 32.
- Configs: dense, and sparse mv:27648 (the C3 128k budget) with `LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`, as in the speed runs.
- Scoring (`port_r_score.py`): RULER's own metrics. string_match_all for niah, vt, cwe and fwe; string_match_part for qa. RULER's postprocessing applies. The score is the mean over the 13 tasks.
- **Gate (non-inferiority):** Δ = sparse − dense ≥ −3.0 points, and the 95% paired bootstrap lower bound ≥ −6.0.
- Prediction: Δ ∈ [−3, +1]. The two tasks with the largest drop are among {cwe, fwe, niah_multikey_3, niah_multivalue, niah_multiquery}, the tasks that need many positions.
- Reported: per-task scores for both configs.

**R2, speed–budget curve at 128k (reported, no gate).**
- 3B with KV f16 and 8B with KV q8_0. `llama-bench -d 131072 -n 64 -r 3`: dense, and sparse at B ∈ {8192, 12288, 16384, 27648 (8B) or 28672 (3B), 32768, 49152}.
- To be read together with the engine KL (C4c, C5a) and retrieval rates (C5b) at the same budgets.

**R3, wall-clock with prefill (reported, no gate).**
- `llama-bench -pg 131072,1024 -r 2`: dense vs sparse at the C3 budget, for the 3B (KV f16) and the 8B (KV q8_0).
- Reported as end-to-end tokens/s and seconds for a 128k prompt plus 1024 generated tokens.
- Expectation, stated now: the end-to-end gain is well below the decode-only S, because prefill is dense and dominates.

Order: R1, then R2, then R3.

## PORT-R amendment 1 (2026-11-05, before any R2/R3 data on either box; all PORT-G results seen)

R2 (speed–budget curve at 128k) and R3 (wall-clock, `-pg 131072,1024`) are also run on the PORT-G RTX 3090 (`vastai13`), with the same commands and budgets. Qwen3-8B (KV q8_0, budgets as for the 8B) is added there. Reported only, no gate, as in PORT-R.

## PORT-R2 (2026-11-06): RULER at larger budgets, and re-validation of the two upstream kernel PRs, on the RTX 4090 (pre-registered before any PORT-R2 data; all PORT-R R1/R2/R3 results, both boxes, and V-UPSTREAM seen)

**Why.** R1 failed at 27648 cells (Δ −7.4, CI [−11.0, −4.1]), with the loss in multi-needle tasks. R2 shows decode is still faster at larger budgets (8B at 128k, 49152 cells: 1.42× on the 3090, 1.24× on the 4090). This test finds whether a budget exists where RULER is non-inferior, which would turn the speed–quality dial into a recommended operating point.

**Box and engine.**
- `vastai11` (RTX 4090 24 GB), the R1 box, restarted with its disk.
- Engine unchanged from R1: llama.cpp `sparse-cuda` 8213b023d, same build, same model file (SHA-256 recorded again), same `ruler8b.i32` token file (SHA-256 recorded) and RULER metadata.
- No code change to the selection path. The proposed page-stats refresh fix is not part of this test.

**R4, RULER at 128k, Llama-3.1-8B q8_0, KV q8_0.**
- Data, prompts, `max_new`, `--gen` mode, env (`LLAMA_SPARSE_SCORE=mv LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`) and scoring (`port_r_score.py`, 10000-sample paired bootstrap): all exactly as R1.
- Configs in one pass (one prefill per sample): dense, mv:49152, mv:65536.
- **Gate per budget (non-inferiority, as R1):** Δ = sparse − dense ≥ −3.0 points, and the 95% paired bootstrap lower bound ≥ −6.0.
- **Recommended operating point:** the smallest tested budget that passes, reported with its measured decode speedup at 128k on both GPUs. If neither passes, there is no recommended budget at 128k for this model; the result is reported as a curve: Δ at 27648, 49152, 65536.
- **Prediction:** Δ(49152) ∈ [−6, −1] (most likely a narrow fail); Δ(65536) ∈ [−4, 0]; Δ is non-decreasing in budget within 1 point. niah_multivalue keeps the largest drop at 49152.
- **Check (no gate):** dense outputs are compared token by token with R1 dense. Any mismatch is reported with its count.
- Reported: per-task scores for all three configs.

**R5, decode speed at 65536 cells (reported, no gate).** `llama-bench -d 131072 -n 64 -r 3`, 8B q8_0 KV, dense and sparse at 49152 and 65536 on the 4090. The 3090 is not re-rented; its 65536 point is reported as predicted by the PORT-G cost model only, labelled as a prediction.

**U, upstream PR re-validation (gates for opening the PRs).**
- Branches as fixed after V-UPSTREAM: `cuda-pool1d-row` 61c176dd8, `cuda-top-k-mask-split` be1b3f418, both on master 6023692b1, built with `CMAKE_CUDA_ARCHITECTURES=89`.
- U1: `test-backend-ops -o POOL_1D -b CUDA0` on `cuda-pool1d-row`: 0 FAIL.
- U2: `test-backend-ops -o TOP_K_MASK -b CUDA0` on `cuda-top-k-mask-split`: 0 FAIL.
- U3: `test-backend-ops perf -o TOP_K_MASK`, median of 3, branch vs master: no case more than 5% slower than master. (V-UPSTREAM found up to 11× slower at nb = 2048 before the fix.)
- U4: `perf -o POOL_1D`, median of 3: no case more than 5% slower than master.
- Also run on `sparse-decode` 394ff1f65: FLASH_ATTN_EXT, POOL_1D and TOP_K_MASK eval, 0 FAIL (reported).
- A PR is opened only if its gates pass.

Order: U (about 30 min), then R5, then R4.

## PORT-R2 amendment 1 (2026-11-06, after U and R5 results, before any R4 data was read)

**Results so far.**
- U1 PASS (224/224). U2 PASS (38/38). U4 PASS (no POOL_1D case slower; 4.3–65× faster). Feature branch FLASH_ATTN_EXT 4148/4148, POOL_1D 216/216, TOP_K_MASK 38/38.
- **U3 FAIL:** decode cases (nb = 1) are 1.2–2.0× faster, but 4 of 8 nb = 2048 cases are 1.06–2.2× slower than master. The `top_k_mask` PR is not opened at be1b3f418.
- R5 (4090, 8B, 128k): 36.9 tok/s dense, 45.5 at 49152 (1.23×), 40.8 at 65536 (1.11×).

**Fix.** `b16c04c05` (PR branch) and `55dee4e39` (feature branch): when `ggml_nrows(dst) >= nsm`, the op runs master's fused one-block-per-row kernel unchanged; only fewer rows take the split path. An eval case with 512 rows is added, so both paths are tested on any GPU.

**U3b (gate for the PR, replaces U3):** on the same box after R4, TOP_K_MASK eval 0 FAIL, and `perf -o TOP_K_MASK` median of 3: no case more than 5% slower than master, and the nb = 1 cases at n_kv 131072 at least 1.3× faster. The feature branch's TOP_K_MASK eval is rerun as well (0 FAIL, reported).

## PORT-L (2026-11-06): decode at 256k context, and whether the quality-safe budget is a fixed number of cells (pre-registered before any PORT-L measurement; setup running; PORT-R R1–R3, PORT-R2 U and R5 seen; R4 running, not read)

**Question.** At 128k the quality-safe budget (R4 will tell which) gives a modest speedup. If what the model needs is a roughly fixed number of cells (the relevant pages plus their neighbourhood) rather than a fixed fraction of the context, the same budget at 256k keeps quality and reads half the fraction, so the speedup grows with context. PORT-L tests this, and tests the cost model 8× beyond its calibration range.

**Box and engine.**
- `vastai14`: RTX 4090 48 GB, CUDA 13.2.
- llama.cpp `sparse-cuda` 2dddd1fa3 = 8213b023d plus microbenchmark cases at 262144 cells and budgets up to 131072 (test-only change).
- Env as in all speed runs: `LLAMA_SPARSE_SCORE=mv LLAMA_SPARSE_PAGE_STATS=f16 GGML_CUDA_FA_SPARSE_PAGED=1`, budget via `LLAMA_SPARSE_PAGES`.
- Model: `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct` (same architecture as Llama-3.1-8B), converted f16 then q8_0 by `setup_port_l.sh`, SHA-256 recorded. KV q8_0.

**B\*.** The smallest R4 budget that passes its gate; 65536 if neither passes. Fixed by R4 before any L2/L3 data.

**L1, speed prediction at 128k and 256k (gate).**
- Phase 1: attention microbenchmarks (`SPARSE_CUDA_PERF`, dense and group-shared g = 4, 3 runs, medians); `llama-bench -n 64` dense at T ∈ {8192, 16384, 32768, 131072, 262144} (10 reps ≤ 32k, 5 above); sparse at the calibration points 8192:2048, 16384:3072, 32768:6144 (10 reps).
- Predictions by `port_g_predict.py` (calibrated at 8k–32k, as PORT-G), for T = 131072 with B ∈ {27648, 49152, 65536} and T = 262144 with B ∈ {27648, 49152, 65536, 98304, 131072}. Committed before Phase 2.
- Phase 2: sparse `llama-bench` at those 8 (T, B) points, 5 reps.
- **Gate:** PASS if all 8 predictions are within ±10% of the measured S; PARTIAL if at least 6.
- **Prediction (stated now, before Phase 1):** S(262144, 65536) ∈ [1.4, 1.9]; S(262144, B) > S(131072, B) for every B in both lists.

**L2, RULER subset at 128k and 256k (the fixed-budget hypothesis).**
- Tasks: the six that lost points in R1: niah_multikey_1–3, niah_multivalue, niah_multiquery, vt. 20 samples per task, seed 42, prepared with RULER db43f92 for each `max_seq_length` (131072 and 262144), the UltraLong tokenizer, and template `meta-llama3` if RULER's template.py has it at that commit, otherwise `base` (recorded). The other R1 tasks are excluded: single-needle tasks were at ceiling, and cwe at the floor.
- Prompts, `max_new` and scoring as R1 (`port_r_tokens.py --tasks`, `port_r_score.py` with the UltraLong tokenizer).
- Configs: 128k: dense, mv:B\*. 256k: dense, mv:B\*, mv:2B\*.
- Δ is the six-task mean, so it is larger in magnitude than a 13-task Δ for the same loss and is not compared to R1/R4 numbers directly.
- **H (fixed absolute budget) is supported** if Δ(256k, B\*) − Δ(128k, B\*) ≥ −3 points (difference of the two paired Δs; 95% CI by independent bootstrap, reported). **H is refuted** if that difference is < −3 and its 95% upper bound is < 0. Otherwise inconclusive.
- Informative only if the dense six-task mean at 256k is ≥ 30; otherwise L2 is reported as uninformative.
- mv:2B\* at 256k (the fixed-fraction alternative) is reported alongside.
- **Prediction:** H supported.

**L3, KL at 128k and 256k (reported, no gate).** `llama-sparse-tail --tokens`, 4 PG19 books per length if the validation and test splits have them (else fewer, reported), 256 teacher-forced tokens, `LLAMA_SPARSE_PAGE_STATS=f32`: dense, mv:B\*, mv:2B\* at each length.

Order: setup, L1 Phase 1, commit predictions, L1 Phase 2, L3, L2 (128k, then 256k).

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

## PORT-L amendment 1 (2026-11-06): B* and page score for Phase 3 (written after R4 and PORT-D D-A were read; before any L2/L3 data; D-B partially run, not read)

**B\* = 49152.** R4 passed at 49152 (Δ −1.91, CI [−4.46, 0.63]) and at 65536; 49152 is the smallest passing budget, so B\* = 49152 and 2B\* = 98304 by the PORT-L rule.

**Page score: quest:d2.** PORT-D D-A (120 prompts, six losing R1 tasks, 27648 cells, stats f16):
- Δ vs dense: mv −15.75 [−22.04, −9.63]; quest −3.38 [−9.08, 2.62]; mv:d2 −18.54 [−25.17, −12.38]; quest:d2 −3.33 [−8.71, 2.08].
- S = +12.38 [6.00, 18.83] (counts); E = −2.79 [−6.04, 0.00] (does not count).
- S counts, so by the PORT-D rule Phase 3 uses the best of the four by Δ: quest:d2. It beats quest by 0.05 points, which is a tie in practice (quest:d2 − quest = +0.04 [−3.12, 3.00]); the rule has no tie-break and is applied as written.
- Checks: D-A dense matches R1 dense token for token on 120/120; D-A mv matches R1 mv token for token on 91/120 (different box and build), with the same score (48.37 vs 48.58).
- Prediction scorecard so far: S counts ✓; E positive but < 3 ✗ (negative); quest:d2 best ✓ (by 0.05); P pending D-B.

**Phase 3 changes.**
- L2 and L3 run with `quest:49152:d2` and `quest:98304:d2` (build 5242a9188), in place of mv.
- Added before L3, so the speed claim matches the configuration whose quality is measured: llama-bench with the same build at 131072 and 262144, 5 reps each: dense, mv:49152 (repeat of a Phase 2 point, as a build check), quest:49152:d2, and at 262144 also quest:98304:d2. Speedup for quest:d2 is reported against the dense run of the same build. No new speed prediction is registered for quest:d2; the L1 predictions (mv) stand as scored.

## PORT-L amendment 2 (2026-11-06): plain quest in Phase 3 (written after PORT-D D-A was read; the first Phase 3 attempt with quest:d2 had run its speed bench at 131072 and part of 262144, not read; no L2/L3 data; D-B finished, not read)

Phase 3 is restarted with `quest` and no dense layers (`quest:49152`, `quest:98304`). Reason: quest:d2 won the PORT-D rule by 0.05 points (quest:d2 − quest = +0.04 [−3.12, 3.00]); the configuration that will be released is plain quest, and the clean feature branch has no dense-layer option. Measuring the released configuration takes priority over the literal tie-break. The bench files of the aborted attempt are kept in `bench/p3_try1_d2/` and reported as such. Everything else in amendment 1 stands, with quest in place of quest:d2.

## PORT-R3 (2026-11-06): full RULER with the quest score at the KL budget (pre-registered after PORT-D D-A; before any R3 data)

**Question.** D-A showed that on the six tasks that lost most in R1, quest at 27648 cells is −3.4 against dense, where mv is −15.7. Does quest pass the R1/R4 RULER gate on all 13 tasks at 27648 cells, the budget of the 128k speed table?

**Setup.** As R1 and R4: `vastai11` (RTX 4090 24 GB), llama.cpp `sparse-cuda` 8213b023d, Llama-3.1-8B q8_0 (SHA-256 checked against R1), KV q8_0, page stats f16, the 260 R1 prompts (`ruler8b.i32`, SHA-256 checked against R1), scoring `port_r_score.py`. Configs: dense, quest:27648 (one process). Speed first: `llama-bench -d 131072 -n 64 -r 3`, dense, mv:27648 and quest:27648.

**Gate (as R1/R4).** Δ = quest − dense ≥ −3.0 points on the 13-task mean, and the 95% paired bootstrap lower bound ≥ −6.0.

**Checks (no gate).** Dense outputs against R1 dense, token by token. quest on the 120 D-A prompts against D-A quest (different box), score and token match reported.

**Predictions.** PASS, with Δ within [−3.0, −0.5]: the six D-A tasks contribute about 6/13 × −3.4 ≈ −1.6, and the other seven were within 5 points each under mv in R1. quest decode speed at 27648 within 3% of mv at 27648.

**Consequence (fixed now).** PASS: the released default is quest, and the 128k speed table at 27648/28672 cells stands as the operating point with RULER measured on it. FAIL: the released default is quest at B\* = 49152, with quality from PORT-L L2 (UltraLong-8B, 128k and 256k); the 27648 point is reported as the faster end of the dial with its R3 RULER score. No further RULER runs either way.
