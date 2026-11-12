<!-- first committed in the project repository as ad168c8 at 2026-11-11T16:13:34+01:00 -->

## PORT-GQ (2026-11-11): the RTX 3090 speed table with the quest score (pre-registered before any PORT-GQ data; PORT-G, R3 quest-vs-mv timing and PORT-RQ seen)

**Question.** The 3090 speed rows (PORT-G) used the mv score; every RULER result that passes uses quest, the released default. quest was timed against mv only once (R3: Llama-3.1-8B, RTX 4090 24 GB, 128k, 51.94 vs 52.05 tok/s). Does quest decode as fast as mv on the RTX 3090, at the PORT-G depths and budgets?

**Engine.** The public release build, `topppg/llama.cpp` `sparse-decode` e7059e52a, `llama-bench`. The run stops if the cloned commit is not e7059e52a. One RTX 3090 (24 GB), `CMAKE_CUDA_ARCHITECTURES=86`.

**Models.** As PORT-G, converted from HF to f16 and quantised to q8_0 with the fork's tools; SHA-256 checked against 068ac283… (Llama-3.2-3B), 38a8de72… (Llama-3.1-8B), 8790136b… (Qwen3-8B-Base, YaRN factor 4). A mismatch is recorded and reported, not a stop.

**Configs.** Per model and depth, in this order: dense, mv, quest (`LLAMA_SPARSE_DECODE=<budget>`, `LLAMA_SPARSE_DECODE_SCORE=mv|quest`, `LLAMA_SPARSE_DECODE_MIN_CTX=0` so sparse runs at every depth, as PORT-G). `llama-bench -fa 1 -p 0 -n 64 -d <depth> -ctk/-ctv <kv>`, 10 repetitions at depth ≤ 32768, 5 above (as PORT-G). Depths and budgets as PORT-G: 3B (f16 KV) 8192=2048, 16384=3072, 32768=4096, 65536=14336, 131072=28672; 8B and Qwen3 (q8_0 KV) 8192=2048, 16384=3072, 32768=6144, 65536=12288, 131072=27648.

**Gate (per model, at 131072).** quest speedup ≥ 0.95 × mv speedup, both against dense on this box.

**Check (no gate).** mv speedup at 131072 on this box against PORT-G (1.35 / 1.77 / 1.81). A difference above 7% is reported as box-to-box variation.

**Prediction.** PASS on all three, with quest within ±3% of mv (R3: −0.2%). quest's score is two dot products against max/min instead of mean/var, so the selection costs the same.

**Consequence (fixed now).**
- The Reddit speed table and chart switch to quest from this run (dense and quest tok/s, speedups), on all three models, pass or fail. The mv/quest caveat and "(mv score)" in the title are removed. If any model fails, the post says quest is slower there and by how much.
- The bench README speed section gets the quest rows from this run, with the PORT-G mv rows kept as history (separate change, after Giorgio's sign-off).
- Published either way.
