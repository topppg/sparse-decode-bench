<!-- first committed in the project repository as 3058b9a at 2026-11-19T14:25:06+01:00 -->

## PF-NT amendment 1 (2026-11-19, 13:30 UTC): box B excluded, its list rerun on box A, degenerate rows

Written after data existed. Condensed from the project repository.

**Seen when written.**
- Box A (vastai38, run `20261119T103525Z`): the status lines of b38 units 1.11, 1.8, 1.0 and 1.1–1.4. Gate F inputs: b38 dense qa_1 70 and vt 100 on pass 1, all 80 outputs direct answers. Gate D: b38 task-0 dense 100, read about 1.5 h after it was ready (late). No combined or sparse score computed.
- Box B (vastai39, run `20261119T103531Z`), q38 units 2.11, 2.8, 2.0, 2.1, 2.2 and 2.3: gate F PASS (all 80 outputs direct answers; dense qa_1 80, vt 90), but **26 of 240 rows degenerate**: 30–32 tokens, all token id 0 (decodes as "!!!!"), always in pairs that share a prefill, so 13 bad prefills out of 120 (11%), in dense and sparse prefills alike.
- Side tests (not PF-NT data), separate `llama-sparse-tail` processes next to the running unit, configs dense and pf, q38: on box B, prompts 8010, 8012 and 11013 with f16 KV gave 1 degenerate prefill of 6 (11013 pf, which was fine in the unit), and the three prefills that had failed in the units were fine; with q8_0 KV 0 of 6. On box A, f16 KV, 10 prompts including all five that failed on box B: 0 of 20.
- Earlier data: 0 degenerate rows in the raw-prompt runs (q38 q8_0 KV and b38 f16 KV, 1,040 rows each) and in box A's 160 b38 rows so far.
- Reading (not established): the failures do not repeat on identical inputs and appear only on box B; a faulty or unstable card is the leading explanation (ECC disabled on both cards). A rare numerical fault of q38 with f16 KV is not excluded: 0 of 20 on box A has probability about 0.1 at box B's rate.

**Changes** (chosen by the project owner at 13:19 UTC):
1. **Box B is excluded.** Its rows are reported descriptively (degenerate counts, scores) and never enter the decision.
2. **Box B's list runs on box A** after box A's first run ends (COMPLETE or FAILED) and is pulled: the registered pass-2 list unchanged (q38 pass 2 in the registered order, then b38 pass 2), in a new run dir on the same box, same cap. Models already present are reused (SHA-256 checked); source, build and gates P, K and U run again. Gates F and D for q38 use the second run's units q38 2.11 + 2.8 and 2.0, as registered. The manifest names the first run dir for pass 1 of both models and the second for pass 2 of both.
3. **Degenerate rows (new quality row 2n, after row 2).** A row is degenerate if it has ≥ 4 generated tokens and at least half of them are token id 0. Any degenerate row among a model's accepted rows makes that model INCONCLUSIVE: such rows measure a numerical failure, not attention quality. Counts are reported per run dir and model.

**Unchanged:** prompts, models, f16 KV for both, configs, gates P, K, U, F and D, floors, the gate, outcomes, predictions. `pfn_decide.py` implements row 2n.
