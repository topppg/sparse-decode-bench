# Scripts

Recompute the main README's numbers from `results/`, or rerun the experiments.

## Recompute from the published data

```
scripts/reproduce.sh            # needs uv and network; no GPU
```

It prints the speed table, the speed–quality dial, wall-clock, the 256k speeds, RULER Δ with 95% CIs for every model and score at 128k and 256k, the task-group table, the mv-failure effects, the 256k test and the cost-model predictions against measurement, and writes the JSON to `reproduced/`. The RULER score files it writes are identical to the committed ones in `results/`. Scoring decodes the generated token ids with four tokenizers downloaded from Hugging Face at pinned revisions.

It then reruns the sparse-prefill decisions (`pf_e2e/decide.sh`, below), checks that they equal the committed decision files, and prints the README's sparse-prefill numbers (`pf_e2e/e2e_summary.py`).

Not recomputed by these scripts: the KL and synthetic-needle checks (produced by `llama-sparse-tail` in the fork; raw outputs in `results/cuda_4090_portc/` and `results/cuda_3090_portg/fid/`), the released-build check (`results/cuda_4090_48gb_featcheck/verdict.json`) and the Metal microbenchmarks.

| script | computes | README section |
|---|---|---|
| `speed_table.py` | tok/s and speedup from llama-bench JSONL pairs; seconds for prompt + generation runs | Speed, dial, wall-clock, 256k |
| `ruler_score.py` | per-task RULER scores, means, Δ vs dense with a paired bootstrap (10 000 resamples); refuses duplicate or unpaired records | Quality, Longer context |
| `ruler_groups.py` | single-needle / multi-needle / other group means | Where the loss is |
| `ruler_effects.py` | paired effects of single changes on the six-task mean | Why mv failed |
| `ruler_context_gate.py` | Δ(256k) − Δ(128k) at a fixed budget, paired bootstrap | Longer context |
| `cost_model.py` | predicted sparse decode speed from dense runs, microbenchmarks and a short-context fit | How the speedup is predicted |
| `multikey_posthoc.py` | answer spans against 64-cell pages, dense vs sparse answers (post hoc) | Qwen3-8B and Llama-3.2-3B |

`multikey_posthoc.py` needs the prompt token file, which is not published (135 MB). Rebuild it as below with the Llama-3.1-8B tokenizer; the run used SHA-256 `e2fb9ecd…2b2` (`results/cuda_4090_portr/sha.txt`).

## Sparse prefill (PF-NT, PF-E2E)

```
scripts/pf_e2e/decide.sh reproduced/pf_e2e      # needs uv and network (the Qwen3.8-27B tokenizer at 37561816); PY=<python> to use another interpreter
cmp reproduced/pf_e2e/decision.json results/pfe2e_analysis/decision.json
cmp reproduced/pf_e2e/decision_as_registered.json results/pfe2e_analysis/decision_as_registered.json
```

- `decide.sh` runs the unchanged decision scripts: `pf_e2e/e2e_decide.py --amend4` (→ `decision.json`) and without `--amend4` (→ `decision_as_registered.json`), and `pf_nt/pfn_decide.py` (→ `pf_nt_decision.json`). These scripts find their helpers and inputs through the project repository's layout (`experiments/src`, `experiments/run_scripts/<protocol>`, `experiments/results/<tag>`), so `decide.sh` rebuilds that layout in a temporary directory from the bench folders, decompresses the `*.err.gz` logs, and runs them from `experiments/src` with the committed runs' relative arguments. Checked: both PF-E2E files are byte-identical to the committed ones, with the project's Python environment and with `uv` (2026-11-23); the PF-NT file is identical except its `vs_raw` fields, which compare with earlier raw-prompt runs that are not published here.
- `pf_e2e/e2e_summary.py` prints the README's sparse-prefill numbers from the decision files and the development run's rows.
- `pf_e2e/src/port_r_score.py`: the RULER scoring functions (`post`, `hit`) the decisions use, copied exactly from the project repository's `port_r_score.py`; they are the same functions as `ruler_score.py`'s.
- `pf_e2e/box/in/`: the box inputs as run: `run_pfe_speed.sh` (speed arms; `ARM=A` RTX 5090, `ARM=D` RTX 4090), `run_pfe_ruler.sh` (RULER units; `UNITS` from `units.env`), `pfe_matrix.tsv` (the timed rows), `e2e_client.py` and `sp_e2e.py` (the client), `e2e_check.py`, `pfe_check.py`, `pfn_check.py` (row and unit checks), `ruler_subset.py`, `sp_lib.sh`, `patchset.env` (trees and upstream commit), and the two patch series `patches/` (0001–0007) and `patches2/` (0001–0008). They apply with `git am` onto fork commit f7d80009d (public, branch `sparse-decode-upstream-20261117` of [topppg/llama.cpp](https://github.com/topppg/llama.cpp)) and give trees 487a7f154 and 9e3e51c75 (checked again on 2026-11-23). `pfe_check.py` is the version after PF-E2E amendment 4 (two lines skip one-token ubatches); the boxes ran the version without them. The rental-provider wrapper that launched these scripts with a hard cap and an automatic stop is not included; the run scripts start without it (`RUN_CAP_S` sets the deadline). The prompt token files are not published; their SHA-256s and metadata are in `results/pfe2e_analysis/inputs/`.
- `pf_nt/`: `pfn_decide.py`, `pfn_gates.py` and the checks in `box/in/`, as used by PF-NT and imported by `e2e_decide.py`. PF-NT's box scripts are not included; its patches are `pf_e2e/box/in/patches/` (the same seven files).
- The scripts carry the project repository's comments, including references to its records (decision and review numbers) that are not in this repository.

## Rerun an experiment

On a machine with an NVIDIA GPU, CUDA and cmake:

```
OUT=ruler_data TOK=Qwen/Qwen3-8B-Base scripts/ruler_prep.sh                 # RULER db43f92, 20 samples per task, seed 42
python scripts/ruler_tokens.py --data ruler_data --model Qwen/Qwen3-8B-Base --out rulerq.i32
scripts/run_ruler.sh qwen3_8b_q8_0.gguf q8_0 rulerq.i32 dense,quest:27648 q_gen.jsonl   # builds the fork at e7059e52a
python scripts/ruler_score.py --gen q_gen.jsonl --meta rulerq.i32.meta.json --model Qwen/Qwen3-8B-Base
```

- Models were converted with llama.cpp's `convert_hf_to_gguf.py` (f16) and quantised to Q8_0 with `llama-quantize`. For Qwen3-8B-Base, `config.json` was first set to YaRN: `"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}`, `"max_position_embeddings": 131072`.
- Model SHA-256 hashes are in the `sha.txt` files under `results/`.
- Speed: the `llama-bench` commands in the main README. Microbenchmarks for the cost model: `test-backend-ops perf -o FLASH_ATTN_EXT` on the fork.
- `ruler_prep.sh` and `run_ruler.sh` are cleaned-up versions of the run scripts used; they were not rerun after the cleanup. The analysis scripts were checked against the committed results.
