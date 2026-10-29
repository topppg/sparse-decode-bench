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
