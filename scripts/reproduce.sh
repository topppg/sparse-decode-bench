#!/bin/bash
# Recompute the README numbers from results/ (no GPU needed; downloads the tokenizers).
# Usage: scripts/reproduce.sh [OUT_DIR]   Needs uv (https://docs.astral.sh/uv/).
set -eu
cd "$(dirname "$0")"
R=../results; O=${1:-$PWD/../reproduced}; mkdir -p "$O"
py() { uv run -q --no-project --python 3.12 --with transformers --with numpy python "$@" > /dev/null 2>&1; }

echo "== Speed table (128k) and 256k"
python3 speed_table.py \
  "RTX 4090 48 GB, 3B=$R/cuda_4090_portc/c3fix/s_d131072_dense.jsonl,$R/cuda_4090_portc/c3fix/s_d131072_sparse.jsonl" \
  "RTX 4090 48 GB, 8B=$R/cuda_4090_portc/c3fix8b/s_d131072_dense.jsonl,$R/cuda_4090_portc/c3fix8b/s_d131072_sparse.jsonl" \
  "RTX 3090, 3B=$R/cuda_3090_portg/bench/3b_d131072_dense.jsonl,$R/cuda_3090_portg/bench/3b_d131072_sparse.jsonl" \
  "RTX 3090, 8B=$R/cuda_3090_portg/bench/8b_d131072_dense.jsonl,$R/cuda_3090_portg/bench/8b_d131072_sparse.jsonl" \
  "RTX 3090, Qwen3-8B=$R/cuda_3090_portg/bench/qwen3_d131072_dense.jsonl,$R/cuda_3090_portg/bench/qwen3_d131072_sparse.jsonl" \
  "quest vs mv, 4090 24 GB, 8B=$R/cuda_4090_portr3/bench/r3_mv27648.jsonl,$R/cuda_4090_portr3/bench/r3_quest27648.jsonl" \
  "UltraLong 128k, 49152=$R/cuda_4090_48gb_portl/bench/p3_d131072_dense.jsonl,$R/cuda_4090_48gb_portl/bench/p3_d131072_quest49152.jsonl" \
  "UltraLong 256k, 49152=$R/cuda_4090_48gb_portl/bench/p3_d262144_dense.jsonl,$R/cuda_4090_48gb_portl/bench/p3_d262144_quest49152.jsonl" \
  "UltraLong 256k, 98304=$R/cuda_4090_48gb_portl/bench/p3_d262144_dense.jsonl,$R/cuda_4090_48gb_portl/bench/p3_d262144_quest98304.jsonl"

echo "== Speed-quality dial (RTX 3090, 128k)"
B3=$R/cuda_3090_portr/bench
for m in 3b 8b qwen3; do
  mid=27648; [ $m = 3b ] && mid=28672
  args=(); for b in 8192 16384 $mid 32768 49152; do args+=("$m, $b cells=$B3/r2_${m}_dense.jsonl,$B3/r2_${m}_b$b.jsonl"); done
  python3 speed_table.py "${args[@]}"
done

echo "== Wall-clock, 128k prompt + 1024 tokens (RTX 3090)"
python3 speed_table.py "3B=$B3/r3_3b_dense.jsonl,$B3/r3_3b_sparse.jsonl" "8B=$B3/r3_8b_dense.jsonl,$B3/r3_8b_sparse.jsonl" "Qwen3-8B=$B3/r3_qwen3_dense.jsonl,$B3/r3_qwen3_sparse.jsonl"

echo "== RULER at 128k (13 tasks)"
L8=unsloth/Meta-Llama-3.1-8B; M8=$R/cuda_4090_portr/ruler8b.i32.meta.json
py ruler_score.py --gen $R/cuda_4090_portr3/r3_gen.jsonl --meta $M8 --model $L8 --out "$O/llama8b_quest.json"
py ruler_score.py --gen $R/cuda_4090_portr/r1_gen.jsonl --meta $M8 --model $L8 --out "$O/llama8b_mv.json"
py ruler_score.py --gen $R/cuda_4090_portr2/r4_gen.jsonl --meta $M8 --model $L8 --out "$O/llama8b_mv_larger.json"
py ruler_score.py --gen $R/cuda_4090_portrq/q_gen.jsonl --meta $R/cuda_4090_portrq/rulerq.i32.meta.json --model Qwen/Qwen3-8B-Base --out "$O/qwen8b_quest.json"
py ruler_score.py --gen $R/cuda_4090_portrq/s_gen.jsonl --meta $M8 --model unsloth/Llama-3.2-3B --out "$O/llama3b_quest.json"
for f in llama8b_quest llama8b_mv llama8b_mv_larger qwen8b_quest llama3b_quest; do
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d.items():
    if k.startswith('delta_'):
        print(f\"{sys.argv[2]:18s} {k[6:]:12s} mean {d['mean']['dense']:.1f} -> {d['mean'][k[6:]]:.1f}  delta {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]\")
" "$O/$f.json" "$f"
done

echo "== Task groups"
python3 ruler_groups.py "$O/llama8b_quest.json" "$O/qwen8b_quest.json" "$O/llama3b_quest.json"

echo "== Why mv failed (six-task effects)"
py ruler_effects.py --meta $M8 --model $L8 --out "$O/mv_effects.json" \
  da=$R/cuda_4090_portd/da_gen.jsonl db=$R/cuda_4090_portd/db_gen.jsonl r1=$R/cuda_4090_portr/r1_gen.jsonl
python3 -c "
import json, sys
e = json.load(open(sys.argv[1]))['effects']
for k, name in (('S', 'quest instead of mv'), ('E', 'first 2 layers dense'), ('P', 'f32 page statistics')):
    print(f'{name:22s} {e[k][0]:+.1f} [{e[k][1][0]:+.1f}, {e[k][1][1]:+.1f}]')
" "$O/mv_effects.json"

echo "== 256k quality test"
py ruler_context_gate.py --dir $R/cuda_4090_48gb_portl/ruler --model nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct --out "$O/context_gate.json"
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(f\"delta 128k {d['delta_128k']:+.1f}, 256k {d['delta_256k']:+.1f}, difference {d['difference']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]: {d['verdict']}\")
" "$O/context_gate.json"

echo "== Cost-model predictions"
uv run -q --no-project --python 3.12 --with numpy python cost_model.py --micro $R/cuda_3090_portg/micro --bench $R/cuda_3090_portg/bench --out "$O/cost_3090.json" \
  --models "3b|3|f16|28|8192:2048,16384:3072,32768:4096,65536:14336,131072:28672;8b|4|q8_0|32|8192:2048,16384:3072,32768:6144,65536:12288,131072:27648;qwen3|4|q8_0|36|8192:2048,16384:3072,32768:6144,65536:12288,131072:27648" > /dev/null
uv run -q --no-project --python 3.12 --with numpy python cost_model.py --micro $R/cuda_4090_48gb_portl/micro --bench $R/cuda_4090_48gb_portl/bench --out "$O/cost_256k.json" \
  --models "ul8b|4|q8_0|32|8192:2048,16384:3072,32768:6144,131072:27648,131072:49152,131072:65536,262144:27648,262144:49152,262144:65536,262144:98304,262144:131072" > /dev/null
python3 -c "
import json, sys
for f in sys.argv[1:]:
    for m, pts in json.load(open(f)).items():
        for k, v in pts.items():
            if isinstance(v, dict) and 'pred_err' in v and not k.startswith(('8192', '16384', '32768')):
                print(f\"{m:6s} T={k:14s} predicted {v['pred_S']:.2f}x  measured {v['S']:.2f}x  error {100 * v['pred_err']:+.1f}%\")
" "$O/cost_3090.json" "$O/cost_256k.json"
echo "JSON outputs in $O"
