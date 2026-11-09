# Scripts

Recompute every number in the main README from `results/`, or rerun the experiments.

## Recompute from the published data

```
scripts/reproduce.sh            # needs uv; no GPU; downloads four tokenizers from Hugging Face
```

It prints the speed table, the speed–quality dial, wall-clock, RULER Δ with 95% CIs for every model and score, the task-group table, the mv-failure effects, the 256k test and the cost-model predictions against measurement, and writes the JSON to `reproduced/`. The RULER score files it writes are identical to the committed ones in `results/`.

| script | computes | README section |
|---|---|---|
| `speed_table.py` | tok/s and speedup from llama-bench JSONL pairs; seconds for prompt + generation runs | Speed, dial, wall-clock, 256k |
| `ruler_score.py` | per-task RULER scores, means, Δ vs dense with a paired bootstrap (10 000 resamples) | Quality |
| `ruler_groups.py` | single-needle / multi-needle / other group means | Where the loss is |
| `ruler_effects.py` | paired effects of single changes on the six-task mean | Why mv failed |
| `ruler_context_gate.py` | Δ(256k) − Δ(128k) at a fixed budget, paired bootstrap | Longer context |
| `cost_model.py` | predicted sparse decode speed from dense runs, microbenchmarks and a short-context fit | How the speedup is predicted |
| `multikey_posthoc.py` | answer spans against 64-cell pages, dense vs sparse answers (post hoc) | Qwen3-8B and Llama-3.2-3B |

`multikey_posthoc.py` needs the prompt token file, which is not published (135 MB). Rebuild it as below with the Llama-3.1-8B tokenizer; the run used SHA-256 `e2fb9ecd…2b2` (`results/cuda_4090_portr/sha.txt`).

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
