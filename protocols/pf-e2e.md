<!-- first committed in the project repository as b9c4085 at 2026-11-21T14:12:56+01:00 -->

## PF-E2E (2026-11-21): one long request end to end at ~260k and ~130k, page-select sparse prefill plus page-sparse decode against the best measured dense, on an RTX 5090 and an RTX 4090, with RULER at 256k (pre-registered before any PF-E2E data)

Condensed from the project repository. Code: `scripts/pf_e2e/` (box inputs in `box/in/`, decision `e2e_decide.py`). The arithmetic in "Predictions" is agent synthesis, not measurement. Reviewed before any data (two rounds of an independent model review); its findings are fixed in this text and the scripts.

**Seen when written.**
- PF-NT (`pf-nt.md`): BOTH_PASS at 128k, f16 KV, chat format, thinking off; q38 −1.65 [−3.25, −0.17].
- Earlier project runs, not in this repository: Qwen3.8 Q4_K_M on raw RULER prompts with q8_0 KV on a 5090 (PF-CONF, −2.38 [−5.00, +0.27]); prefill alone through the earlier gather path on a 5090 at 500 W (PF-SPEED: q38 dense q8_0 71.08 s, gather 43.06 s); `llama-bench` at depth 261120 on a 5090 (PORT-CM: q38 dense 34.30 tok/s, quest:27648 66.16 tok/s); an end-to-end run at 128k with MTP (PORT-SP).
- Two development runs (not data): one on the earlier gather path with server checks and nsys, one with the page-list patches (tree 9e3e51c75: bit-identical logits on 3 prompts, sparse-tail prefill 1.336×, server 1.300×). Read after the dev data: their 1,000-token outputs were " France\n" repeated, because `ignore_eos` continues a short RULER answer, so their MTP acceptance (0.973) and output-identity checks were measured on repetition.
- A post hoc fit of the earlier 128k per-prompt prefill times against length (5090, q8_0, 180 rows each, 115,661–131,037 tokens): dense 2.351e-4·T + 2.322e-9·T² s; pf (GATHER=1) 2.688e-4·T + 4.103e-10·T² s. Used only in "Predictions".
- Design choices recorded in the project repository before this protocol: an Amdahl argument that the prefill gain grows with length (the n² share of prefill); RULER at 256k with 13 tasks × 20 on two 5090s alongside the speed box; best dense to include upstream master and `-ub 2048` dense; a registered 128k speed-only arm on an RTX 4090; a book-summary prompt for timing; a 5090 at ≥ 500 W accepted, power recorded and not gated.

**Question.**
1. For one request of ~260k prompt tokens plus 1,000 generated tokens through `llama-server` on an RTX 5090 32 GB (Qwen3.8-27B UD-Q4_K_M, q8_0 KV), how much less wall time does block-sparse prefill through the page-select gather (`LLAMA_SPARSE_PREFILL_GATHER=2`, selective q8_0 conversion) plus page-sparse decode (quest:27648:f3) take than the best measured dense (fork and upstream, `-ub 512` and `-ub 2048`, with and without MTP)?
2. The same at ~130k on the same card: is the gain larger at 256k?
3. The same at ~130k on an RTX 4090 24 GB (speed only), which is also the registered measurement for an earlier stopping rule for sparse prefill on that card (see the speed decision).
4. Does the combined config pass the RULER gate at 256k, 13 tasks × 20 prompts, chat format with thinking off, q8_0 KV?

**Targets.** Qwen3.8-27B, released 2026-08-05; a survey on 2026-11-20 found it still the most-downloaded GGUF (7.38M), no newer consumer NVIDIA card, and no training-free sparse prefill or decode on CUDA upstream. Cards: RTX 5090 32 GB (newest consumer NVIDIA card); RTX 4090 24 GB (the card of most 24 GB users). Architecture (`Qwen/Qwen3.8-27B` config.json at 37561816): 64 layers, 16 full attention (every 4th), 48 Gated DeltaNet; 4 KV heads, head dim 256; native context 262,144; 1 MTP layer. KV of the full-attention layers: 34,816 B/token at q8_0 (8,704 MiB at 262,144), 65,536 B/token at f16 (16,384 MiB at 262,144).

**Memory** (agent arithmetic from the dev logs, scaled linearly from 132,096 cells on the 4090; 32,109 MiB visible on the 5090): at 262,144, dense q8_0 ~26.1 GB (margin ~6.0), pf + sparse decode ~27.8 GB (~4.3), MTP dense ~29.7 GB (~2.4), `-ub 2048` dense ~27–28.5 GB (unmeasured compute buffer); dense f16 KV ~32.7–33.2 GB and pf + MTP > 32 GB: not run. An OOM is a registered outcome, not a failure.

**Engine.**
- Fork: topppg/llama.cpp f7d80009d + PF patches 0001–0007 (`box/in/patches/`, tree **487a7f15479d39eac0b3bf4be0ab46160eca0fb7**) + page-list patches 0001–0008 (`box/in/patches2/`, tree **9e3e51c7579772fa60d6d8b1688c9bf09e54d307**), applied by `git am` on a fresh clone; both trees and a clean work tree checked on the box. Checked locally by applying the patch files to f7d80009d (both trees reproduced).
- Upstream: ggml-org/llama.cpp master **c6a25fc622d322b9ad6e079e97009f7892543a4c** (2026-11-20 head), `llama-server` only.
- Build: CUDA arch from the card; Release; fork `test-backend-ops llama-sparse-tail llama-server`, upstream `llama-server`.
- Server, every config: `-ngl 99 -fa on -np 1 -b 2048 -ub <512|2048> -ctk <kv> -ctv <kv> --cache-ram 0 --ctx-checkpoints 0 -c <132096|262144>`; MTP: `--spec-type draft-mtp -ctkd <kv> -ctvd <kv>`. Graph reuse at the default (the fork never reuses sparse-prefill graphs). `--ctx-checkpoints 0` is needed by the sparse prefill and applied to every config.
- Environment: every process starts from a cleared environment; fork servers get `GGML_CUDA_FA_SELECT_LOG=1`; sparse prefill `LLAMA_SPARSE_PREFILL=0.1 LLAMA_SPARSE_PREFILL_WINDOW=512 LLAMA_SPARSE_PREFILL_ROWS=0:(n−513) LLAMA_SPARSE_PREFILL_GATHER=2 GGML_CUDA_FA_SELECT_CONVERT=1` (the sparse rows of `llama-sparse-tail`'s `d512`, which prefills n−1 tokens); sparse decode `LLAMA_SPARSE_DECODE=<B> LLAMA_SPARSE_DECODE_MIN_CTX=32768 LLAMA_SPARSE_DECODE_FUSED=3`. The dense tail is set from the known prompt length: a prototype control, not a server feature.
- `llama-sparse-tail` (gate B, references, RULER): `--kv-type q8_0`, `-ub 512`, `LLAMA_GRAPH_REUSE_DISABLE=1`, `LLAMA_SPARSE_PREFILL_GATHER=2` (gate B also 1), `GGML_CUDA_FA_SELECT_CONVERT=1`, `GGML_CUDA_FA_SELECT_LOG=1`.

**Model.** `unsloth/Qwen3.8-27B-GGUF` 4af51b35, `Qwen3.8-27B-UD-Q4_K_M.gguf`, SHA-256 1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529.

**Timed prompt** (`pfe_book_q38.i32`, SHA-256 **d66f088a2a37bf4e96f82ebb738944526b6756f35110c975e464e92addaf0bb6**, meta in `results/pfe2e_analysis/inputs/`):
- Text: PG19 validation books 1 and 2 (4 × 131,072 Qwen3 tokens in the source file), decoded with the Qwen3 tokenizer, joined by a blank line, re-encoded with Qwen3.8's (263,805 tokens). Book 0 starts in Gutenberg licence text and is not used.
- id 256: the first 259,500 tokens of that document; id 128: the first 129,500 (a string prefix of the same text). Each is the user turn `<document>\n\nSummarize the text above in detail, section by section.` in Qwen3.8's chat template with thinking off.
- Lengths **259,525** (server `-c 262144`) and **129,526** (`-c 132096`); `max_new` 1000. encode(decode(ids)) = ids on both.
- Why a book and not RULER: RULER answers repeat under `ignore_eos`, which inflates MTP and trivialises identity checks; prefill and sparse-decode cost do not depend on content; a summary request is the natural 256k use.

**RULER prompts at 256k** (`pfe_ruler256_q38.i32`, SHA-256 **c37d000b7714911a3b0195d8e8d5e5b7c82a813db048de2f0a70aa24e8113c58**, meta and provenance in `results/pfe2e_analysis/inputs/`):
- NVIDIA/RULER db43f92, 13 tasks, 20 samples per task, seed 42, base template, `--max_seq_length 261632`, tokenizer `Qwen/Qwen3.8-27B` at 37561816; RULER inputs with the SHA-256s in `ruler256_inputs.sha256`.
- Chat format with thinking off as PF-NT; every prompt gains exactly 12 tokens. Round trips: as ids on all 260; as text up to the tokenizer's NFC normaliser (one prompt, 12016 of qa_2).
- 260 prompts, lengths 222,523–261,600; longest + `max_new` = 261,644 ≤ 262,144. Greedy, RULER's generation lengths (niah 128, vt 30, cwe 120, fwe 50, qa 32).
- **qa_1 at 256k repeats documents** (found in the local prep): SQuAD dev has 1,204 distinct contexts, so at this length RULER's `qa.py` repeats them, and every qa_1 sample holds the same 1,477 documents (273 duplicated) in the same order; only the question differs. RULER builds it so by design. Registered handling: the primary Δ is over the 13 tasks as RULER builds them; the 12-task Δ without qa_1 is reported next to it, descriptive. If the gate outcome differs between them, the verdict says so.

**Boxes and arms.**

| box | card | arm | registered order | cap (from launch, setup included) |
|---|---|---|---|---|
| A | RTX 5090 32 GB | speed, 256k then 128k | `pfe_matrix.tsv` arm A, file order | 18,000 s |
| B | RTX 5090 32 GB | RULER pass 1 (prompts 0–9) | `UNITS_B` in `units.env` | 54,000 s |
| C | RTX 5090 32 GB | RULER pass 2 (prompts 10–19) | `UNITS_C` | 54,000 s |
| D | RTX 4090 24 GB | speed, 128k only | `pfe_matrix.tsv` arm D, file order | 10,800 s |

- **Gate P:** A, B, C: card name contains "RTX 5090", memory.total 32,000–33,000 MiB; D: "RTX 4090", 24,000–25,000 MiB (a 48 GB modified card is refused). Disk ≥ 80 GB with ≥ 60 GB free minus the model file present; RAM ≥ 60 GiB (A, B, C; see amendment 1), ≥ 30 GiB (D). Power limit recorded, not gated.
- A box that fails gate P is replaced by one of the same card running the same list; a missing RULER pass is rerun by amendment on a new box, that pass's list only.

**Run, speed arms** (`run_pfe_speed.sh`, ARM=A|D).
1. Matrix file checked and hashed; gate P; prompt file SHA-256; the two prompts extracted (`PROMPTS OK 256:259525 128:129526`).
2. Fork source (both trees, `SOURCE OK`), upstream at the pinned commit (`UPSTREAM OK`), both builds; model SHA-256 (`MODEL OK`).
3. **Gate K** (fork, `test-backend-ops`): PF-NT's four filters (`n_top`, `n_kv_max=[1-9]`, `n_pages`, the sparse op list; all pass, count > 0), the page-select cases `-p materialise` with `GGML_CUDA_FA_SELECT_CONVERT` 1, 0 and 2 (exactly 24 each), every `FLASH_ATTN_EXT` case; all pass. `SETUP DONE`; nvidia-smi logs every second.
4. Per length: **gate B** and the **reference**: `llama-sparse-tail --gen` on the prompt with 64 tokens, configs `dense`, `pf:0.1:512:d512`, `+quest:27648:f3`, `+quest:55296:f3`; rows checked; the exit line gives S, the selective-conversion count of one pf prefill (`REF <len> OK`).
5. **Timed matrix** (`pfe_matrix.tsv`, SHA-256 b901deded0d6e423a125dae8179a8fa1e57427fbebb85efb63b5a796f86a0cc4):
   - A: 256k block 1: `fork_dense`, `fork_dense_ub2048`, `fork_mtp`, `up_dense`, `up_mtp`, `pfsd27648`, `pfsd55296`, `pf_only`; 256k block 2: the reverse; then 128k blocks 1 and 2 the same; then informative, one block: `sd27648` at 256k; `sd27648`, `fork_dense_f16`, `fork_mtp_f16`, `pfsd27648_f16` at 128k.
   - D: 128k blocks 1 and 2 as A's; informative `sd27648`.
   - Per row: a fresh server on a random port; `/props` must report the model file, n_ctx and one slot; one untimed smoke request (the first 512 prompt tokens, 64 generated); then **2 timed requests**: the full prompt, `n_predict` 1000, greedy, `ignore_eos`, `cache_prompt: false`, timed at the client (`e2e_client.py`, `sp_e2e.py`).
   - `e2e_check.py srv` classifies each row: **OK** (2 valid timed rows: every prompt token processed, exactly 1,000 output ids, timings present and positive, MTP drafted > 0; and the count rule), **OOM** (an out-of-memory message in the server log; an OOM in block 1 skips that config in block 2), **INVALID**. Count rule: a fork server with sparse prefill and q8_0 KV prints "2·S selective, 0 whole" at exit; any other fork server prints no count line or "0 selective, 0 whole". Rows are shape-checked before any arithmetic; a malformed row is INVALID.
   - Deadline (launch + cap − 600 s): a row is admitted only if its estimate fits; one that ends past the deadline makes the run FAILED "deadline" and is not recorded.
6. Final line COMPLETE (every row OK or OOM), INCOMPLETE <items>, or FAILED <reason>.

Estimated: A ~4.1 h, D ~1.6–1.9 h.

**Run, RULER arm** (`run_pfe_ruler.sh`, boxes B and C). Setup and gate K as the speed arm. **Gate A:** on the box's first task-0 prompt (B: id 0, C: id 10), configs `dense,quest:262144:f3` (every page selected) give identical greedy outputs, else FAILED. Units in the registered order: one `llama-sparse-tail --gen` process over 10 prompts × configs `dense`, `quest:27648:f3`, `pf:0.1:512:d512`, `pf:0.1:512:d512+quest:27648:f3`, `pf:0.1:512:d512+quest:55296:f3` (two shared prefills per prompt), validated by `pfn_check.py task` (coverage, counts, length ≥ 100,000, sparse engaged on 16 layers, no reused prefill graph, timings) and `e2e_check.py ruler-sel` (exactly one exit count line, selective > 0, whole 0), then committed atomically. One retry on a crash or missing rows; FAILED on invalid rows, a second failure or the deadline. Unit timeout min(6,000 s, time left). Status lines `FORMAT READY q38` after the second unit (gate F), `TASK0 READY q38` after the task-0 unit (gate D), `Q38 UNITS COMPLETE` after the last; final COMPLETE or FAILED. Estimated ~12.4 h per box.

**Gates.**
- **P, K, trees, SHA-256s:** a failure stops the box (FAILED).
- **B (bit identity, speed boxes):** `llama-sparse-tail` tail mode on the timed prompt at each length, 32 teacher-forced steps, configs `dense,pf:0.1:512:d512`, once with GATHER=1 and once with GATHER=2 (CONVERT=1): the saved pf logits must be byte-identical. A failure is recorded and the run goes on. Meaning: the measured path computes exactly what PF-NT and PF-CONF measured at 128k (they ran GATHER=1); at 256k the gate links the two paths and validates neither.
- **Counts:** every `llama-sparse-tail` row as expected (16 layers, 4 KV heads, no fallback, dense rows only in the tail); the server conversion-count rule (q8_0 sparse rows only). A failed count makes that row INVALID.
- **Decode engagement** (decision script): the server has no sparse-decode counter, so a config with sparse decode is INVALID in a block where its decode tok/s is below 1.30 × `fork_dense`'s (`pfsd27648` at 256k) or 1.15 × (every other case).
- **A (RULER boxes):** above.
- **F** (format): box B's units 1.11 and 1.8, 20 dense outputs: no `<think>`, at most 2 empty or blank (PF-NT's rule). Box C's units 2.11 and 2.8: descriptive.
- **D:** box B's task-0 unit dense ≥ 80; box C's descriptive. A failure seen live stops B and C.
- **N** (natural output): `fork_dense`'s first timed output at each length has ≥ 50% distinct 8-grams; else every MTP row is labelled "on degenerate or unscreened text" (best dense still includes MTP).
- **Degenerate RULER rows** (PF-NT amendment 1: ≥ 4 generated tokens, half or more token id 0): any makes the quality decision INCONCLUSIVE.
- **Informative, never a gate:** S; the first 64 tokens of each server output against its `llama-sparse-tail` reference (the server prefills the last prompt token while `llama-sparse-tail` decodes it, so exact equality is not expected); determinism of each config's four timed outputs; drift; peak memory.used per row.

**Speed decision** (`e2e_decide.py`).
- Measured: client wall time of one request (prompt + 1,000 generated tokens). Also per row: TTFT (`prompt_ms`), decode tok/s, MTP acceptance.
- Arm INVALID if a setup line is missing, there is no terminal line, the matrix hash differs from the registered one, or the prompt lengths differ from the meta.
- Per (length, block, config): median of the 2 timed requests, classification recomputed from the files.
- **Best dense** of a block: the lowest median wall among the OK dense candidates (`fork_dense`, `fork_dense_ub2048`, `fork_mtp`, `up_dense`, `up_mtp`); none if any candidate is INVALID or missing; an OOM candidate is left out.
- **Quoted ratio** R_L(config) = min over blocks 1 and 2 of (best dense wall / config wall); none if a block lacks a best dense or the config is not OK in both. Computed for `pfsd27648` (primary), `pfsd55296` and `pf_only`. Also reported: TTFT ratio against the same best dense and against `fork_dense`, decode ratio, drift (block 2 / block 1, flagged beyond ±3%).
- **Length effect** (5090, each sparse config): R_256 − R_128; **GROWS** iff ≥ 0.3, else NOT_SHOWN. The ±3% drift rule lets a block ratio near 2 move by about 0.06, and both lengths take the lower of two blocks, so 0.3 is about five times that noise and below the predicted difference (~0.65).
- **Stopping rule on arm D** (a rule recorded earlier in the project repository for sparse prefill on the RTX 4090, applied here to a registered measurement): `fork_dense` TTFT / `pfsd27648` TTFT at 128k, lower of the two blocks. ≥ 1.30: cleared; < 1.30: the rule applies (sparse prefill work stops and is written up as a negative speed result with its quality result).
- **Arm D's quality basis**, stated wherever its number appears, only if gate B passed on that card: "quality rests on the bit identity of GATHER=2 with GATHER=1 (gate B) plus PF-NT and PF-CONF at 128k; no RULER was run on this card." Otherwise: "inherited quality unavailable (gate B <FAIL|missing>)".

**Quality decision at 256k** (`e2e_decide.py`, manifest `{"q38": {"1": <box B run dir>, "2": <box C run dir>}}`, first match):

| row | condition | quality |
|---|---|---|
| 1 | a manifest dir lacks SETUP DONE, SOURCE OK, a GATE K line, GATE A PASS or Q38 UNITS COMPLETE, or has no terminal line | INVALID → NO_DECISION |
| 2 | cohort not exact (13 × 20 × 5, each (id, config) once, counts valid), a unit's selective-count line missing or wrong, or any degenerate row | INCONCLUSIVE → NO_DECISION |
| 3 | gate F fails (inputs available) / unavailable | FORMAT_FAIL / NO_DECISION |
| 4 | gate D fails, or the dense 13-task mean < 60 | UNINFORMATIVE |
| 5 | `pf+quest:27648` passes: Δ ≥ −3 and 95% lower bound ≥ −6 | PASS_27648 |
| 6 | else `pf+quest:55296` passes | PASS_55296 |
| 7 | else `pf:0.1:512:d512` alone passes | PASS_PF_ONLY |
| 8 | otherwise | FAIL_ALL |

Δ = mean over the 13 tasks of each task's mean paired difference (config − dense) × 100; bootstrap 10,000, seed 0, one resample of the 20 prompt indices per task; scoring as PF-NT. No per-task floors (no 256k reference exists). Rows 5–7 are a registered sequence of fallbacks, not three chances at one claim: a claim on 55296 or on pf alone says it is the fallback after the primary failed, and the primary's result is reported in the same place. Reported, never changing the outcome: every config's stats on both passes and per pass, per-task scores and deltas, empty outputs, kinds, the 12-task Δ without qa_1, gates F and D on box C, dense per task against PF-NT's 128k dense (descriptive).

**Predictions** (agent arithmetic; ranges fixed now).
- Arithmetic: dense prefill ≈ a·n + b·n²; from the earlier fit the n² share is 56% at 129.5k and 72% at 259.5k on the 5090 (4090 nsys split: 41% and 58%). 5090 dense prefill ~69.4 s at 129.5k, ~218 s at 259.5k. pf prefill on the earlier path 97.6 s at 259.5k; with the page-select gain taken as 5–10%, 88–93 s. Decode at 256k (`llama-bench`): dense 34.3 tok/s (29.2 s for 1,000 tokens), sparse 66.2 tok/s (15.1 s). 256k request: dense ~247 s; pf + sparse decode ~103–108 s, 2.3–2.4× over `fork_dense`; MTP dense ~252–260 s; `-ub 2048` dense 204–236 s if its prefill is 5–20% faster. 128k, 5090: dense ~90 s, sparse 52–54 s. 128k, 4090: dense ~91 s, pf + sparse decode ~70 s.
- P1: gates P, K, B (both lengths, both speed boxes), counts and A pass; no row INVALID.
- P2: MTP dense fits at 262,144 on the 5090 (p ≈ 0.8); `-ub 2048` dense fits there (p ≈ 0.8); peaks within ±1.5 GB of the memory table.
- P3: 5090, 256k: `fork_dense` TTFT 190–250 s (point 218); `pfsd27648` TTFT 80–110 s (point 90); TTFT ratio 1.9–2.8.
- P4: 5090, 256k decode through the server: dense 30–36 tok/s; `pfsd27648` 55–68 tok/s (ratio 1.7–2.0); MTP dense 45–70 tok/s.
- P5: **R_256 (5090) 1.7–2.6, point 2.2.**
- P6: **R_128 (5090) 1.35–1.80, point 1.55.**
- P7: R_256 − R_128 ≥ 0.3.
- P8: the best dense at 256k is a non-MTP config (p ≈ 0.6); `fork_dense_ub2048` is the best dense in at least one block at each length (p ≈ 0.6).
- P9: R_256(pfsd55296) is 0.03–0.12 below R_256(pfsd27648); R_256(pf_only) is 0.15–0.45 below it.
- P10: **R_128 (4090) 1.10–1.35, point 1.22**; the arm D TTFT ratio 1.25–1.40, cleared (≥ 1.30) with p ≈ 0.55.
- P11: RULER at 256k: dense 13-task mean 70–92; `pf+quest:27648` Δ −5 to +1, PASS_27648 with p ≈ 0.55; the 55296 Δ at or above the 27648 Δ − 0.5; qa_1's dense score at 256k below its 128k score (PF-NT q38: 85).
- P12: gate N passes on the book prompt (p ≈ 0.7); MTP acceptance there 0.55–0.85.

**Consequence (fixed now).** Speed and quality are published either way, as measured. The **claim config** comes from the quality outcome first (PASS_27648 → `pfsd27648`, PASS_55296 → `pfsd55296`, PASS_PF_ONLY → `pf_only`; otherwise `pfsd27648` for the speed-only rows); only then is its own R_256 required. Its **gain band** describes the absolute 256k gain over the best measured dense, not a length effect: HIGH (R_256 ≥ 1.8), MODERATE (1.4–1.8), LOW (< 1.4).

| quality at 256k | row | consequence |
|---|---|---|
| PASS_27648 | PASS_27648_<gain> | A public statement may say: one ~260k request R_256× faster than the best dense we measured in llama.cpp on a 5090, with RULER Δ at 256k. |
| PASS_55296 | PASS_55296_<gain> | The same with `pfsd55296` and its own ratio, saying that the primary config did not pass the gate at 256k (non-inferiority not shown, not a demonstrated loss) and the claim uses the registered fallback. |
| PASS_PF_ONLY | PASS_PF_ONLY_<gain> | A 256k statement for sparse prefill alone with R_256(pf_only), saying that neither combined config passed. |
| FAIL_ALL | NO_QUALITY_CLAIM | No 256k quality claim; the speed of `pfsd27648` is reported next to the gate that was not passed; the 128k statements stand. |
| UNINFORMATIVE, FORMAT_FAIL, NO_DECISION, not run | SPEED_ONLY | 256k is speed only; quality stated as measured at 128k, the gap named. NO_DECISION: fix the cause, one rerun of the affected pass by amendment. |
| any, the claim config without a quoted R_256 | NO_RATIO | Fix the cause; one rerun of the affected length by amendment. |

By gain band: HIGH: no further kernel work for speed on this model. MODERATE: reported as measured; no headline beyond the number. LOW: the sparse-prefill program ends at the 128k write-up plus this result. **Length wording, separate:** "the gain grows with context" is allowed only if the claim config's own length effect is GROWS; otherwise R_256 and R_128 are reported side by side, and with NOT_SHOWN the text says the length argument is not supported on this card at the registered margin. Arm D: R_128 (4090) is reported with its quality basis sentence and the stopping-rule outcome; if the rule triggers, it applies whatever arm A shows.

**What it settles.** On one 5090 and one 4090, one model file, one book prompt at two lengths: the wall time and TTFT of the combined path against the best measured dense (fork and upstream at pinned commits, ub 512 and 2048, ±MTP); whether the gain grows from 128k to 256k; the registered 4090 number; RULER quality of the combined config at 256k with q8_0 KV in the chat format.

**What it does not settle.** Other cards (24 GB cards cannot hold 256k for this model, estimated), f16 KV at 256k (does not fit), other weights, models and thinking-on use; generation lengths other than 1,000; serving with batches, prompt caching and multi-turn use; the quality of the 1,000-token summary (not scored); MTP with sparse decode; sparse prefill at `-ub` > 512; a server-side dense tail without the known prompt length; q8_0 KV with the chat format at 128k; RULER quality on the 4090; whether a dedicated kernel would add more.

**Local checks before any box (no GPU).** Prompt files built and hashed; RULER inputs match the earlier runs' hashes; qa_1's document repetition found and handled; the patch files applied to f7d80009d reproduce both trees; stubbed dry runs of both box scripts, the checks, the decision and the live helper on synthetic data (88 tests, 0 failures); the default scoring path run once on synthetic rows.
