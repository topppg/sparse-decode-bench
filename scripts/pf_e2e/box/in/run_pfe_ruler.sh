#!/bin/bash
# PF-E2E RULER arm (experiments/PROTOCOLS.md, PF-E2E): RULER at 256k (pfe_ruler256_q38.i32: 13 tasks x 20 prompts in
# Qwen3.8's no-thinking chat format), Qwen3.8-27B UD-Q4_K_M, KV q8_0, -ub 512, on one RTX 5090 32 GB per pass. PF-NT's
# run_pf_nt.sh with one model, five configs per prompt (two shared prefills), the page-select gather
# (LLAMA_SPARSE_PREFILL_GATHER=2, GGML_CUDA_FA_SELECT_CONVERT=1) and the pf-lists source; gate U (nsys) is replaced by the
# per-unit selective-conversion count, and gate A (quest:262144 = dense) runs first.
# Units in env UNITS (required, no default): "q38:<pass>.<task> ...", pass 1|2, task 0..12, each at most once, one pass per
# box, run exactly in the given order. Registered orders (units.env), cap 54000 s per box, pull window 1200 s:
#   B (UNITS_B): q38 1.11 1.8 1.0 1.1-1.7 1.9 1.10 1.12      C (UNITS_C): the same tasks in pass 2
#   a rerun of a missing pass, by amendment on a new box and run dir: that pass's list alone.
# Launch only through e2e_wrap.sh:
#   source /root/pfe2/in/units.env; UNITS="$UNITS_B" setsid nohup bash /root/pfe2/in/e2e_wrap.sh /root/pfe2/in/run_pfe_ruler.sh 54000 1200 > /root/pfe2_wrap.log 2>&1 < /dev/null &
# /root/pfe2/in/ (rsynced from the Mac): box/in/* plus pfe_ruler256_q38.i32 (results/pf_e2e/inputs/; SHA-256 checked).
# Gate P: the card name contains "RTX 5090"; memory.total >= 32000 MiB; disk size >= 80 GB, free >= 60 GB minus the model
#   file present; RAM >= 28 GiB (amendment 1). The power limit is recorded, not gated.
# Gate K: run_pfe_speed.sh's (PF-NT's four filters, the page-select cases with CONVERT 1, 0, 2, every FLASH_ATTN_EXT case).
# Gate A, after SETUP DONE and before the first unit: the box's first task-0 prompt (pass 1: id 0, pass 2: id 10), configs
#   dense,quest:262144:f3 (every page selected): greedy outputs identical, else FAILED "gate A".
# Units (pass p, task t): pass 1 = ids t*1000 + 0..9, pass 2 = t*1000 + 10..19; one llama-sparse-tail process over the 10
#   prompts x 5 configs, validated (pfn_check.py task: coverage, counts, len >= 100000, sparse engaged on 16 layers, no
#   reused prefill graph, timings; then e2e_check.py ruler-sel: one exit count line, selective > 0, whole 0) and committed
#   atomically into ruler/q38_p<p>_t<t>.jsonl (+ .err, .check); one retry on a crash or missing rows; FAILED on invalid
#   rows, a second crash, or the deadline. A unit's timeout is min(6000 s, the time left to the deadline); a unit that ends
#   past the deadline is FAILED "deadline", not committed.
# Status: "FORMAT READY q38" after the second unit (lead: gate F), "TASK0 READY q38" after the task-0 unit (lead: gate D),
#   "Q38 UNITS COMPLETE" after the last unit (this box's units exact); final line COMPLETE or FAILED <reason>. A TERM writes
#   "FAILED killed by TERM ...". Lead control: touch $R/STOP stops the run (FAILED) before the next unit.
set -u
[ -n "$(echo ${UNITS:-})" ] || { echo "UNITS is empty: give the box's registered list (units.env: UNITS_B or UNITS_C)"; exit 1; }
declare -A SEEN=(); NU=0; PASSES=""
for u in $UNITS; do
  [[ "$u" =~ ^q38:[12]\.([0-9]|1[0-2])$ ]] && [ -z "${SEEN[$u]:-}" ] || { echo "bad or duplicated unit '$u' in UNITS='$UNITS'"; exit 1; }
  SEEN[$u]=1; NU=$((NU+1)); p=${u#q38:}; PASSES="$PASSES ${p%.*}"
done
[ $(echo $PASSES | tr ' ' '\n' | sort -u | wc -l) = 1 ] || { echo "UNITS mixes passes: one pass per box"; exit 1; }
UNITS=$(echo $UNITS); LASTU=${UNITS##* }; MUNITS=$(for u in $UNITS; do echo -n "${u#q38:} "; done)
PASS=$(echo $PASSES | cut -d' ' -f1)
W=${SP_ROOT:-/root/pfe2}; I=$W/in
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}; R=$W/out/$RUN_ID
grep -qs "^SETUP START" $R/status.txt && { echo "run dir $R already used"; exit 1; }
SRC=$R/src; B=$SRC/build/bin
[ -e $SRC ] && { echo "source dir $SRC exists: a fresh checkout is required"; exit 1; }
mkdir -p $R/ruler $R/tmp
COMMIT=f7d80009d4293728489cd2b961fccffa4982bad4
source $I/patchset.env
RUN_CAP_S=${RUN_CAP_S:-54000}; T0=$(date +%s); DEADLINE=$(( T0 + RUN_CAP_S - 600 ))
UNIT_TO=6000                                         # per-unit timeout, s
EST=3600                                             # s per unit until one is measured, then the max measured
MEASURED=0
export PATH=/usr/local/cuda/bin:$PATH
U38=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4af51b3542f775b0a8e4b310bfb3f17e01c306c9
declare -A FILE=([q38]=Qwen3.8-27B-UD-Q4_K_M.gguf) URL=([q38]=$U38) SHA=([q38]=1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529)
TOKF=$I/pfe_ruler256_q38.i32
TSHA=${PFE_RULER_SHA:?PFE_RULER_SHA unset: patchset.env must name the RULER token file}
Q=quest:27648:f3; Q2=quest:55296:f3; PF=pf:0.1:512:d512
CFG=dense,$Q,$PF,$PF+$Q,$PF+$Q2
source $I/sp_lib.sh
trap 'st "FAILED killed by TERM (hard cap or operator)"; exit 143' TERM
PF_VARS="LLAMA_SPARSE_PREFILL LLAMA_SPARSE_PREFILL_WINDOW LLAMA_SPARSE_PREFILL_MIN_CTX LLAMA_SPARSE_PREFILL_ROWS LLAMA_GRAPH_REUSE_DISABLE LLAMA_SPARSE_PREFILL_GATHER GGML_CUDA_PDL GGML_CUDA_FA_SELECT_CONVERT GGML_CUDA_FA_SELECT_LOG"
for v in $PF_VARS; do unset $v; DENSE_ENV="$DENSE_ENV -u $v"; done
# sparse-tail sets the prefill and decode controls per config itself; the page-select gather with selective conversion is
# on for every call, PDL at its default
TAIL_ENV="$DENSE_ENV LLAMA_GRAPH_REUSE_DISABLE=1 LLAMA_SPARSE_PREFILL_GATHER=2 GGML_CUDA_FA_SELECT_CONVERT=1 GGML_CUDA_FA_SELECT_LOG=1"
tail_run() { local to=$1; shift; grp $to $TAIL_ENV $B/llama-sparse-tail -m $(m q38) --kv-type q8_0 "$@"; }
fits() { [ $(( $(date +%s) + $1 )) -le $DEADLINE ]; }
# build concurrency bounded by memory (REV-032): min(nproc, MemAvailable / 2 GiB), at least 1, recorded; sp_lib's build()
# calls $(nproc), which this function shadows, so sp_lib.sh stays byte-identical to PORT-SP's
MEMINFO=${MEMINFO:-/proc/meminfo}
NPROC_ALL=$(command nproc); MEMAV_KB=$(awk '/^MemAvailable:/{print $2}' $MEMINFO 2>/dev/null)
if [ -n "$MEMAV_KB" ]; then JOBS=$(( MEMAV_KB / 2097152 )); [ $JOBS -lt 1 ] && JOBS=1; [ $JOBS -gt $NPROC_ALL ] && JOBS=$NPROC_ALL
else JOBS=$NPROC_ALL; fi
nproc() { echo $JOBS; }
subset() {  # pass task out: the unit's 10 prompts, checked exactly
  local p=$1 t=$2 o=$3 sel
  rm -f $o
  if [ $p = 1 ]; then sel="--per-task 10 --tasks $t"
  else sel="--per-task 20 --tasks $t --ids $(seq $((t*1000+10)) $((t*1000+19)) | paste -sd, -)"; fi
  python3 $I/ruler_subset.py $TOKF $o $sel > /dev/null && python3 $I/pfn_check.py subset $o --task $t --pass $p > /dev/null
}

st "SETUP START run $RUN_ID pass $PASS units \"$UNITS\" ($NU) (fork tree ${PATCH2_TREE:0:9}, deadline +$(( DEADLINE - T0 ))s)"
nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,power.default_limit,power.max_limit,enforced.power.limit,driver_version,compute_cap,clocks.max.sm,clocks.max.mem --format=csv > $R/gpu.txt 2>&1
{ df -h /; command nproc; free -g; nvcc --version | tail -1; lscpu | grep -E "Model name|^CPU\(s\)"; } >> $R/gpu.txt 2>&1
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
case "$NAME" in *"RTX 5090"*) ;; *) fail "gate P: card is $NAME, not an RTX 5090";; esac
[ "${MEM:-0}" -ge 32000 ] || fail "gate P: memory.total $MEM MiB is not a 32 GB card"
PRESENT=$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]) if os.path.isfile(sys.argv[1]) else 0)' $(m q38))
DNEED=$(( 60 - PRESENT / 1073741824 ))
DSIZE=$(df -BG --output=size / | tail -1 | tr -dc 0-9); DAVAIL=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "${DSIZE:-0}" -ge 80 ] && [ "${DAVAIL:-0}" -ge $DNEED ] || fail "gate P: disk ${DSIZE:-?} GB, ${DAVAIL:-?} GB free (need >= 80 and >= $DNEED)"
RAM=$(free -g | awk '/^Mem:/{print $2}')
[ "${RAM:-0}" -ge 28 ] || fail "gate P: RAM ${RAM:-?} GiB < 28"
st "GATE P PASS $NAME ${MEM} MiB power.limit ${PL:-?} W (recorded, not gated) disk ${DSIZE} GB (${DAVAIL} free, need $DNEED) RAM ${RAM} GiB"
[ "$(sha256sum $TOKF 2>/dev/null | tee -a $R/sha.txt | cut -d' ' -f1)" = $TSHA ] || fail "sha tokens pfe_ruler256_q38"
subset 1 0 $R/tmp/s1.i32 || fail "subset pass 1 task 0"
subset 2 0 $R/tmp/s2.i32 || fail "subset pass 2 task 0"
AID=$([ $PASS = 1 ] && echo 0 || echo 10)
python3 $I/ruler_subset.py $TOKF $R/tmp/p_a.i32 --per-task 20 --tasks 0 --ids $AID > $R/tmp/p_a.txt || fail "subset gate A"
grep -q -- "-> 1 prompts" $R/tmp/p_a.txt || fail "subset gate A: $(cat $R/tmp/p_a.txt)"
command -v aria2c > /dev/null || { apt-get update -qq && apt-get install -y -qq aria2 > /dev/null; } || fail "apt aria2"
dl q38 & BG="$BG $!"; DL=$!
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
CMK="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_USE_PREBUILT_UI=OFF"

fetch_fork
git -C $SRC -c user.name=box -c user.email=box@localhost am -q $I/patches/*.patch || fail "git am patches"
TREE=$(git -C $SRC rev-parse HEAD^{tree})
[ "$TREE" = "$PATCH_TREE" ] || fail "tree after patches $TREE != $PATCH_TREE"
git -C $SRC -c user.name=box -c user.email=box@localhost am -q $I/patches2/*.patch || fail "git am patches2"
TREE2=$(git -C $SRC rev-parse HEAD^{tree}); git -C $SRC log --oneline -16 >> $R/commit.txt
[ "$TREE2" = "$PATCH2_TREE" ] || fail "tree after patches2 $TREE2 != $PATCH2_TREE"
[ -z "$(git -C $SRC status --porcelain --untracked-files=all)" ] || fail "source tree not clean after git am"
st "SOURCE OK tree $TREE2 clean, $(ls $I/patches/*.patch $I/patches2/*.patch | wc -l | tr -d " ") patches on ${COMMIT:0:9}"
st "BUILD JOBS $JOBS (nproc $NPROC_ALL, MemAvailable ${MEMAV_KB:-unknown} kB)"
build $SRC $SRC/build fork test-backend-ops llama-sparse-tail || fail "build"
[ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || fail "tracked sources changed during the build"
st "BUILD DONE"

tbo_gate() {  # tag expected-count(0 = any > 0) env-assignments op [filter]
  local tag=$1 want=$2 ev=$3 op=$4 line n mm f=(); shift 4
  [ $# -gt 0 ] && f=(-p "$1")
  grp 3600 env $ev $B/test-backend-ops test -b CUDA0 -o $op "${f[@]}" > $R/tbo_$tag.txt 2>&1 || { echo "rc=$?"; return 1; }
  line=$(grep -oE '[0-9]+/[0-9]+ tests passed' $R/tbo_$tag.txt | tail -1); n=${line%%/*}; mm=${line#*/}; mm=${mm%% *}
  [ -n "$line" ] && [ "$n" -gt 0 ] && [ "$n" = "$mm" ] && ! grep -q "FAIL" $R/tbo_$tag.txt || { echo "'${line:-no test count}'"; return 1; }
  [ "$want" = 0 ] || [ "$n" = "$want" ] || { echo "$n cases, expected $want"; return 1; }
  echo "$line"
}
for g in "blocks 0 - FLASH_ATTN_EXT n_top" "sparse 0 - FLASH_ATTN_EXT n_kv_max=[1-9]" "pages 0 - FLASH_ATTN_EXT n_pages" \
         "ops 0 - TOP_K_MASK,PAGE_TOP_K,ARGSORT,POOL_1D,PAD,SOFT_MAX,CPY,REPEAT,CONCAT,SCALE,ADD,CONT,RELU,NEG" \
         "select_c1 24 GGML_CUDA_FA_SELECT_CONVERT=1 FLASH_ATTN_EXT materialise" "select_c0 24 GGML_CUDA_FA_SELECT_CONVERT=0 FLASH_ATTN_EXT materialise" \
         "select_c2 24 GGML_CUDA_FA_SELECT_CONVERT=2 FLASH_ATTN_EXT materialise" "fa_all 0 - FLASH_ATTN_EXT"; do
  set -f; set -- $g; set +f; ev=$3; [ "$ev" = - ] && ev=""
  res=$(tbo_gate $1 $2 "$ev" $4 ${5:+"$5"}) || fail "gate K $1: $res"
  st "GATE K $1 PASS $res"
done
wait $DL; shaok q38 || fail "sha q38"
st "SETUP DONE"
grp_bg nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv -l 5 > $R/gpu_log.csv 2>&1

# gate A: with a budget above the cache every page is selected, so the greedy output equals dense (decode hooks at 256k)
mkdir -p $R/gate_a
to=$(( DEADLINE - $(date +%s) )); [ $to -gt 2400 ] && to=2400; [ $to -gt 0 ] || fail "deadline: no time left for gate A"
tail_run $to --gen $R/tmp/p_a.i32 --configs dense,quest:262144:f3 > $R/gate_a/rows.jsonl 2> $R/gate_a/err.txt || fail "gate A run rc=$?"
[ $(date +%s) -lt $DEADLINE ] || fail "deadline: gate A ended past the deadline"
res=$(python3 $I/e2e_check.py gate-a $R/gate_a/rows.jsonl) || fail "gate A: $res"
st "GATE A $res"

unit() {  # pass task
  local p=$1 t=$2 f=$R/ruler/q38_p$1_t$2 tmp=$R/tmp/q38_p$1_t$2 rc d t1=$(date +%s) to
  [ -e $f.jsonl ] && fail "duplicate ruler $f"
  rm -f $tmp.*
  subset $p $t $R/tmp/r.i32 || { st "RULER q38 p$p task $t subset failed"; return 1; }
  to=$(( DEADLINE - $(date +%s) )); [ $to -gt $UNIT_TO ] && to=$UNIT_TO
  [ $to -gt 0 ] || fail "deadline: pass $p task $t has no time left before the deadline"
  tail_run $to --gen $R/tmp/r.i32 --configs $CFG > $tmp.jsonl 2> $tmp.err; rc=$?
  [ $(date +%s) -lt $DEADLINE ] || fail "deadline: pass $p task $t ended past the deadline (rc $rc, timeout $to s), not committed"
  [ $rc = 0 ] || { st "RULER q38 p$p task $t run rc=$rc"; return 1; }
  python3 $I/pfn_check.py task $tmp.jsonl --model q38 --task $t --pass $p --configs $CFG --err $tmp.err > $tmp.check 2>&1; rc=$?
  [ $rc = 0 ] || { st "RULER q38 p$p task $t $(tail -1 $tmp.check | cut -c1-300)"; [ $rc = 1 ] && return 1; return 2; }
  python3 $I/e2e_check.py ruler-sel $tmp.err >> $tmp.check 2>&1 || { st "RULER q38 p$p task $t $(tail -1 $tmp.check)"; return 2; }
  [ $(date +%s) -lt $DEADLINE ] || fail "deadline: pass $p task $t validated past the deadline, not committed"
  mv $tmp.err $f.err && mv $tmp.check $f.check && mv $tmp.jsonl $f.jsonl || fail "commit $f"
  d=$(( $(date +%s) - t1 ))
  if [ $MEASURED = 0 ] || [ $d -gt $EST ]; then EST=$d; fi; MEASURED=1
  st "RULER q38 p$p task $t done in $d s (pf/dense prefill median $(grep '^TASK' $f.check | grep -oE '[0-9.]+$|n/a$'); $(tail -1 $f.check))"
}

i=0; FIRSTF=""
for u in $UNITS; do
  i=$((i+1)); pt=${u#q38:}; p=${pt%.*}; t=${pt#*.}
  [ -e $R/STOP ] && fail "stopped by the lead (gate F or D) before pass $p task $t"
  fits $EST || fail "deadline: pass $p task $t cannot finish before the deadline (estimate $EST s)"
  unit $p $t; rc=$?
  if [ $rc = 1 ]; then
    fits $EST || fail "deadline: no time to retry pass $p task $t"
    st "RULER q38 p$p task $t retry"; unit $p $t; rc=$?
  fi
  [ $rc = 0 ] || fail "pass $p task $t (rc $rc, see status)"
  [ $i = 1 ] && FIRSTF=ruler/q38_p${p}_t${t}.jsonl
  [ $i = 2 ] && st "FORMAT READY q38 $FIRSTF ruler/q38_p${p}_t${t}.jsonl: gate F on the Mac; fails -> touch $R/STOP on both boxes"
  [ $t = 0 ] && st "TASK0 READY q38 ruler/q38_p${p}_t0.jsonl: gate D on the Mac; fails -> touch $R/STOP on both boxes"
  if [ $u = $LASTU ]; then
    python3 $I/pfn_check.py cohort $R/ruler --model q38 --configs $CFG --units "$MUNITS" > $R/ruler/cohort_q38.json 2>&1 ||
        fail "q38 units not exact: $(head -c 300 $R/ruler/cohort_q38.json)"
    st "Q38 UNITS COMPLETE ($NU units) $(head -c 200 $R/ruler/cohort_q38.json)"
  fi
done
kill $BG 2>/dev/null
finish
