#!/bin/bash
# Build the released fork and generate RULER answers, dense and sparse, with llama-sparse-tail --gen.
# The prompt cache is filled densely once per prompt; each config then decodes greedily from it.
# Usage: run_ruler.sh MODEL.gguf KV_TYPE PROMPTS.i32 CONFIGS OUT.jsonl
#   e.g. run_ruler.sh qwen3_8b_q8_0.gguf q8_0 rulerq.i32 dense,quest:27648 q_gen.jsonl
# Env: LLAMA_DIR (default ./llama.cpp-sparse), COMMIT (default e7059e52a). Needs CUDA and cmake.
set -eu
[ $# -eq 5 ] || { sed -n 2,7p "$0"; exit 1; }
MODEL=$1; KV=$2; PROMPTS=$3; CONFIGS=$4; OUTF=$5
L=${LLAMA_DIR:-$PWD/llama.cpp-sparse}; COMMIT=${COMMIT:-e7059e52a}
[ -d "$L" ] || git clone -q -b sparse-decode https://github.com/topppg/llama.cpp "$L"
git -C "$L" checkout -q "$COMMIT"
if [ ! -x "$L/build-cuda/bin/llama-sparse-tail" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  cmake -B "$L/build-cuda" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release "$L" > cmake.log 2>&1
  cmake --build "$L/build-cuda" -j "$(nproc)" --target llama-sparse-tail > build.log 2>&1
fi
git -C "$L" rev-parse HEAD > "$OUTF.commit"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUTF.gpu"
"$L/build-cuda/bin/llama-sparse-tail" -m "$MODEL" --kv-type "$KV" --gen "$PROMPTS" --configs "$CONFIGS" > "$OUTF" 2> "$OUTF.err"
wc -l < "$OUTF"
