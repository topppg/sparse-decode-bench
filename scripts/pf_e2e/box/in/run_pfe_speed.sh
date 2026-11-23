#!/bin/bash
# PF-E2E speed arm (experiments/PROTOCOLS.md, PF-E2E): one long request end to end (the book prompt, 1,000 generated
# tokens) through a fresh llama-server per (length, block, config), block-sparse prefill through the page-select gather
# (LLAMA_SPARSE_PREFILL_GATHER=2, selective q8_0 conversion) plus page-sparse decode, against the best measured dense
# (fork and upstream, -ub 512 and 2048, with and without MTP). Qwen3.8-27B UD-Q4_K_M, KV q8_0 (f16 rows informative).
#   ARM=A: RTX 5090 32 GB, lengths 256 (prompt id 256, -c 262144) then 128 (id 128, -c 132096), cap 18000 s
#   ARM=D: RTX 4090 24 GB, length 128 only, cap 10800 s
# Launch only through e2e_wrap.sh (prerequisite checks, independent absolute-deadline stopper, hard cap, pull window):
#   ARM=A setsid nohup bash /root/pfe2/in/e2e_wrap.sh /root/pfe2/in/run_pfe_speed.sh 18000 1200 > /root/pfe2_wrap.log 2>&1 < /dev/null &
# /root/pfe2/in/ (rsynced from the Mac): box/in/* plus pfe_book_q38.i32 (results/pf_e2e/inputs/; SHA-256 checked).
# Stages (status.txt lines in caps):
#   SETUP START; GATE P (card name contains "RTX 5090" for A / "RTX 4090" for D, memory.total 32000-33000 / 24000-25000 MiB,
#     disk >= 80 GB with >= 60 GB free minus the model file present, RAM >= 28 / 30 GiB (amendment 1); power limit recorded, not gated);
#   prompt file SHA-256; SOURCE OK (fork f7d80009d + patches/ -> tree PATCH_TREE, + patches2/ -> PATCH2_TREE, clean);
#   UPSTREAM OK (ggml-org/llama.cpp at UP_COMMIT, clean); BUILD DONE fork / upstream; GATE K (fork: PF-NT's four filters,
#   the page-select cases -p materialise with GGML_CUDA_FA_SELECT_CONVERT 1, 0 and 2, exactly 24 each, and every
#   FLASH_ATTN_EXT case); MODEL OK; SETUP DONE;
#   per length: GATE B <len> PASS|FAIL (llama-sparse-tail tail mode on the prompt, 32 teacher-forced steps,
#     configs dense,pf:0.1:512:d512, GATHER=1 against GATHER=2, CONVERT=1: saved pf logits byte-identical; a FAIL is
#     recorded and the run goes on); REF <len> (sparse-tail --gen, 64 tokens, configs dense, pf, pf+quest:27648,
#     pf+quest:55296, GATHER=2: counts checked, the process's selective conversion count S for that prompt);
#   the matrix rows of this arm (pfe_matrix.tsv, file order): E2E <len> b<block> <config> OK|OOM|INVALID|SKIP ...
#     (e2e_check.py srv: two valid timed rows, the conversion-count rule; an OOM in block 1 skips that config in block 2);
#   final line COMPLETE (every row OK or OOM), INCOMPLETE <items>, or FAILED <reason>; TERM writes INCOMPLETE (sp_lib).
# Deadline (launch + cap - 600 s): the timeouts of gate B, the references and every server row are min(their own, the
#   time left), and a row is admitted only if its estimate fits; one of those that ends (or a row whose validation ends)
#   past the deadline is FAILED "deadline" (a row's files renamed *.late). Builds (5400 s per step, sp_lib) and gate K
#   (3600 s per filter) keep fixed timeouts; they run before SETUP DONE, inside the wrapper's hard cap.
# Build concurrency: min(nproc, MemAvailable / 2 GiB), recorded as BUILD JOBS.
# Server flags, every config: -ngl 99 -fa on -np 1 -b 2048 -ub <512|2048> -ctk/-ctv <q8_0|f16> --cache-ram 0
#   --ctx-checkpoints 0 -c <132096|262144>; MTP: --spec-type draft-mtp -ctkd/-ctvd <kv> (the GGUF's MTP layer); graph reuse
#   at the default. Every process starts from a cleared environment; fork configs get GGML_CUDA_FA_SELECT_LOG=1; sparse
#   prefill: LLAMA_SPARSE_PREFILL=0.1 _WINDOW=512 _ROWS=0:(n-513) _GATHER=2 GGML_CUDA_FA_SELECT_CONVERT=1 (the rows of
#   sparse-tail's d512 on the n-1 prefilled tokens); sparse decode: LLAMA_SPARSE_DECODE=<B> _MIN_CTX=32768 _FUSED=3.
set -u
ARM=${ARM:-}
case "$ARM" in A|D) ;; *) echo "ARM must be A (RTX 5090, 256k and 128k) or D (RTX 4090, 128k)"; exit 1;; esac
W=${SP_ROOT:-/root/pfe2}; I=$W/in
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}; R=$W/out/$RUN_ID
grep -qs "^SETUP START" $R/status.txt && { echo "run dir $R already used"; exit 1; }
SRC=$R/src; B=$SRC/build/bin; USRC=$R/src_up; BU=$USRC/build/bin
{ [ -e $SRC ] || [ -e $USRC ]; } && { echo "source dirs exist in $R: a fresh checkout is required"; exit 1; }
mkdir -p $R/e2e $R/tmp $R/bit $R/ref
COMMIT=f7d80009d4293728489cd2b961fccffa4982bad4      # fork base (sparse-decode-upstream-20261117)
source $I/patchset.env                               # PATCH_TREE, PATCH2_TREE, UP_COMMIT
RUN_CAP_S=${RUN_CAP_S:-18000}; T0=$(date +%s); DEADLINE=$(( T0 + RUN_CAP_S - 600 ))
export PATH=/usr/local/cuda/bin:$PATH
U38=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4af51b3542f775b0a8e4b310bfb3f17e01c306c9
declare -A FILE=([q38]=Qwen3.8-27B-UD-Q4_K_M.gguf) URL=([q38]=$U38) SHA=([q38]=1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529)
TOKF=$I/pfe_book_q38.i32
TSHA=${PFE_BOOK_SHA:?PFE_BOOK_SHA unset: patchset.env must name the book token file}   # src/pfe2e_tokens.py book
source $I/sp_lib.sh
PF_VARS="LLAMA_SPARSE_PREFILL LLAMA_SPARSE_PREFILL_WINDOW LLAMA_SPARSE_PREFILL_MIN_CTX LLAMA_SPARSE_PREFILL_ROWS LLAMA_GRAPH_REUSE_DISABLE LLAMA_SPARSE_PREFILL_GATHER GGML_CUDA_PDL GGML_CUDA_FA_SELECT_CONVERT GGML_CUDA_FA_SELECT_LOG"
for v in $PF_VARS; do unset $v; DENSE_ENV="$DENSE_ENV -u $v"; done
PF=pf:0.1:512:d512
REFCFG=dense,$PF,$PF+quest:27648:f3,$PF+quest:55296:f3
TAIL_ENV() { echo "$DENSE_ENV LLAMA_GRAPH_REUSE_DISABLE=1 LLAMA_SPARSE_PREFILL_GATHER=$1 GGML_CUDA_FA_SELECT_CONVERT=1 GGML_CUDA_FA_SELECT_LOG=1"; }
case $ARM in
  A) GPU_PAT="RTX 5090"; MEM_MIN=32000; MEM_MAX=33000; RAM_MIN=28; LENS="256 128";;
  D) GPU_PAT="RTX 4090"; MEM_MIN=24000; MEM_MAX=25000; RAM_MIN=30; LENS="128";;
esac
declare -A CTX=([128]=132096 [256]=262144) NLEN=() SELREF=() EST=([128]=400 [256]=900) OOMED=()
N_PREDICT=1000; REPS=2; E2E_TO=3600
fits() { [ $(( $(date +%s) + $1 )) -le $DEADLINE ]; }
bto() {  # a stage timeout: min(<s>, the time left to the deadline); 0 when none is left
  local t=$(( DEADLINE - $(date +%s) )); [ $t -gt $1 ] && t=$1; [ $t -gt 0 ] && echo $t || echo 0
}
past() { [ $(date +%s) -ge $DEADLINE ]; }
# build concurrency bounded by memory (REV-032): min(nproc, MemAvailable / 2 GiB), at least 1, recorded; sp_lib's build()
# calls $(nproc), which this function shadows, so sp_lib.sh stays byte-identical to PORT-SP's
MEMINFO=${MEMINFO:-/proc/meminfo}
NPROC_ALL=$(command nproc); MEMAV_KB=$(awk '/^MemAvailable:/{print $2}' $MEMINFO 2>/dev/null)
if [ -n "$MEMAV_KB" ]; then JOBS=$(( MEMAV_KB / 2097152 )); [ $JOBS -lt 1 ] && JOBS=1; [ $JOBS -gt $NPROC_ALL ] && JOBS=$NPROC_ALL
else JOBS=$NPROC_ALL; fi
nproc() { echo $JOBS; }

st "SETUP START run $RUN_ID arm $ARM lengths \"$LENS\" (fork tree ${PATCH2_TREE:0:9}, upstream ${UP_COMMIT:0:9}, deadline +$(( DEADLINE - T0 ))s)"
python3 $I/e2e_check.py matrix $I/pfe_matrix.tsv --arm $ARM > $R/matrix.txt || fail "matrix: $(head -c 300 $R/matrix.txt)"
sha256sum $I/pfe_matrix.tsv >> $R/sha.txt
nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,power.default_limit,power.max_limit,enforced.power.limit,driver_version,compute_cap,clocks.max.sm,clocks.max.mem --format=csv > $R/gpu.txt 2>&1
{ df -h /; command nproc; free -g; nvcc --version | tail -1; lscpu | grep -E "Model name|^CPU\(s\)"; } >> $R/gpu.txt 2>&1
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
case "$NAME" in *"$GPU_PAT"*) ;; *) fail "gate P: card is $NAME, not an $GPU_PAT (arm $ARM)";; esac
[ "${MEM:-0}" -ge $MEM_MIN ] && [ "${MEM:-0}" -le $MEM_MAX ] || fail "gate P: memory.total ${MEM:-?} MiB outside $MEM_MIN-$MEM_MAX (arm $ARM)"
PRESENT=$(python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]) if os.path.isfile(sys.argv[1]) else 0)' $(m q38))
DNEED=$(( 60 - PRESENT / 1073741824 ))
DSIZE=$(df -BG --output=size / | tail -1 | tr -dc 0-9); DAVAIL=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
[ "${DSIZE:-0}" -ge 80 ] && [ "${DAVAIL:-0}" -ge $DNEED ] || fail "gate P: disk ${DSIZE:-?} GB, ${DAVAIL:-?} GB free (need >= 80 and >= $DNEED)"
RAM=$(free -g | awk '/^Mem:/{print $2}')
[ "${RAM:-0}" -ge $RAM_MIN ] || fail "gate P: RAM ${RAM:-?} GiB < $RAM_MIN"
st "GATE P PASS $NAME ${MEM} MiB power.limit ${PL:-?} W (recorded, not gated) disk ${DSIZE} GB (${DAVAIL} free, need $DNEED) RAM ${RAM} GiB"
[ "$(sha256sum $TOKF 2>/dev/null | tee -a $R/sha.txt | cut -d' ' -f1)" = $TSHA ] || fail "sha tokens pfe_book_q38"
for L in $LENS; do
  NLEN[$L]=$(python3 $I/e2e_check.py prompt $TOKF --id $L --max-new $N_PREDICT --out $R/tmp/p_$L.i32 --raw $R/tmp/raw_$L.i32) || fail "prompt $L"
  python3 $I/e2e_check.py prompt $TOKF --id $L --max-new 64 --out $R/tmp/p64_$L.i32 > /dev/null || fail "prompt $L (64)"
  [ $(( ${NLEN[$L]} + N_PREDICT )) -le ${CTX[$L]} ] || fail "prompt $L: ${NLEN[$L]} + $N_PREDICT > ${CTX[$L]}"
done
st "PROMPTS OK $(for L in $LENS; do echo -n "$L:${NLEN[$L]} "; done)"
command -v aria2c > /dev/null || { apt-get update -qq && apt-get install -y -qq aria2 > /dev/null; } || fail "apt aria2"
dl q38 & BG="$BG $!"; DL=$!
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
CMK0="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$ARCH -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_USE_PREBUILT_UI=OFF"

fetch_fork
git -C $SRC -c user.name=box -c user.email=box@localhost am -q $I/patches/*.patch || fail "git am patches"
TREE=$(git -C $SRC rev-parse HEAD^{tree})
[ "$TREE" = "$PATCH_TREE" ] || fail "tree after patches $TREE != $PATCH_TREE"
git -C $SRC -c user.name=box -c user.email=box@localhost am -q $I/patches2/*.patch || fail "git am patches2"
TREE2=$(git -C $SRC rev-parse HEAD^{tree}); git -C $SRC log --oneline -16 >> $R/commit.txt
[ "$TREE2" = "$PATCH2_TREE" ] || fail "tree after patches2 $TREE2 != $PATCH2_TREE"
[ -z "$(git -C $SRC status --porcelain --untracked-files=all)" ] || fail "fork tree not clean after git am"
st "SOURCE OK fork tree $TREE2 clean ($(ls $I/patches/*.patch $I/patches2/*.patch | wc -l | tr -d " ") patches on ${COMMIT:0:9})"
mkdir -p $USRC && git -C $USRC init -q && git -C $USRC remote add origin https://github.com/ggml-org/llama.cpp.git &&
  git -C $USRC fetch -q --depth 1 origin $UP_COMMIT && git -C $USRC checkout -q --detach FETCH_HEAD || fail "upstream fetch"
[ "$(git -C $USRC rev-parse HEAD)" = $UP_COMMIT ] && [ -z "$(git -C $USRC status --porcelain --untracked-files=all)" ] || fail "upstream tree"
git -C $USRC rev-parse HEAD > $R/commit_upstream.txt
st "UPSTREAM OK $UP_COMMIT clean"
st "BUILD JOBS $JOBS (nproc $NPROC_ALL, MemAvailable ${MEMAV_KB:-unknown} kB)"
CMK="$CMK0 -DLLAMA_BUILD_TESTS=ON"
build $SRC $SRC/build fork test-backend-ops llama-sparse-tail llama-server || fail "build fork"
st "BUILD DONE fork"
CMK="$CMK0 -DLLAMA_BUILD_TESTS=OFF"
build $USRC $USRC/build upstream llama-server || fail "build upstream"
[ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] && [ -z "$(git -C $USRC status --porcelain --untracked-files=no)" ] ||
    fail "tracked sources changed during the builds"
st "BUILD DONE upstream"

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
st "MODEL OK"
st "SETUP DONE"
grp_bg nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv,noheader -l 1 > $R/gpu_log.csv 2>&1
echo "setup_done $(date +%s)" >> $R/stages.txt

# gate B and the references, per length
for L in $LENS; do
  n=${NLEN[$L]}
  fits $(( 4 * ${EST[$L]} )) || fail "deadline: no time for gate B and the reference at $L"
  for g in 1 2; do
    to=$(bto 3600); [ $to -gt 0 ] || fail "deadline: no time left for gate B $L gather $g"
    grp $to $(TAIL_ENV $g) $B/llama-sparse-tail -m $(m q38) --kv-type q8_0 --tokens $R/tmp/raw_$L.i32 --T $n --n-tail 32 \
        --configs dense,$PF --save-logits $PF=$R/tmp/lg_${L}_g$g.bin > $R/bit/tail_${L}_g$g.jsonl 2> $R/bit/tail_${L}_g$g.err ||
        fail "gate B $L gather $g run rc=$?"
    past && fail "deadline: gate B $L gather $g ended past the deadline"
  done
  sha256sum $R/tmp/lg_${L}_g1.bin $R/tmp/lg_${L}_g2.bin | sed "s#$R/tmp/##" >> $R/bit/bin_sha256.txt
  if [ -s $R/tmp/lg_${L}_g1.bin ] && cmp -s $R/tmp/lg_${L}_g1.bin $R/tmp/lg_${L}_g2.bin; then
    st "GATE B $L PASS pf logits of GATHER=2 byte-identical to GATHER=1 ($(wc -c < $R/tmp/lg_${L}_g1.bin | tr -d ' ') bytes, 32 steps)"
  else
    st "GATE B $L FAIL pf logits differ: $(cmp $R/tmp/lg_${L}_g1.bin $R/tmp/lg_${L}_g2.bin 2>&1 | head -1)"
  fi
  rm -f $R/tmp/lg_${L}_g*.bin
  to=$(bto 3600); [ $to -gt 0 ] || fail "deadline: no time left for the reference at $L"
  grp $to $(TAIL_ENV 2) $B/llama-sparse-tail -m $(m q38) --kv-type q8_0 --gen $R/tmp/p64_$L.i32 --configs $REFCFG \
      > $R/ref/gen_$L.jsonl 2> $R/ref/gen_$L.err || fail "reference $L rc=$?"
  past && fail "deadline: the reference at $L ended past the deadline"
  SELREF[$L]=$(python3 $I/e2e_check.py ref $R/ref/gen_$L.jsonl --err $R/ref/gen_$L.err --len $n --configs $REFCFG) ||
      fail "reference $L: ${SELREF[$L]}"
  st "REF $L OK len $n, selective conversion count per prefill ${SELREF[$L]}"
done

srv_case() {  # len block config
  local L=$1 blk=$2 c=$3 n=${NLEN[$1]} bn=$2 envs="$DENSE_ENV" bin out log v rc t1 d extra xd=""
  [ $blk = i ] && bn=0
  eval "$(python3 $I/e2e_check.py cfg $c)" || fail "config $c"
  if [ $blk = 2 ] && [ -n "${OOMED[$L.$c]:-}" ]; then st "E2E $L b$blk $c SKIP (OOM in block 1)"; return 0; fi
  fits ${EST[$L]} || fail "deadline: $L b$blk $c cannot finish before the deadline (estimate ${EST[$L]} s)"
  bin=$B/llama-server; [ $C_BIN = up ] && bin=$BU/llama-server
  extra="-ub $C_UB -ctk $C_KV -ctv $C_KV"
  [ $C_MTP = 1 ] && { extra="$extra --spec-type draft-mtp -ctkd $C_KV -ctvd $C_KV"; xd="--expect-draft"; }
  [ $C_BIN = fork ] && envs="$envs GGML_CUDA_FA_SELECT_LOG=1"
  [ $C_PF = 1 ] && envs="$envs LLAMA_SPARSE_PREFILL=0.1 LLAMA_SPARSE_PREFILL_WINDOW=512 LLAMA_SPARSE_PREFILL_ROWS=0:$((n-513)) LLAMA_SPARSE_PREFILL_GATHER=2 GGML_CUDA_FA_SELECT_CONVERT=1"
  [ $C_SD != 0 ] && envs="$envs LLAMA_SPARSE_DECODE=$C_SD LLAMA_SPARSE_DECODE_MIN_CTX=32768 LLAMA_SPARSE_DECODE_FUSED=3"
  out=$R/e2e/${L}_b${blk}_$c.jsonl; log=$R/e2e/srv_${L}_b${blk}_$c.log
  [ -e $out ] && fail "duplicate e2e $out"
  SPORT=$(( 20000 + RANDOM % 20000 )); t1=$(date +%s)
  echo "${L}_b${blk}_${c}_start $t1" >> $R/stages.txt
  setsid $envs $bin -m $(m q38) -ngl 99 -fa on -c ${CTX[$L]} -np 1 -b 2048 $extra --cache-ram 0 --ctx-checkpoints 0 \
      --host 127.0.0.1 --port $SPORT > $log 2>&1 &
  SPID=$!; echo $SPID >> $PGF
  to=$(bto $E2E_TO); [ $to -gt 0 ] || { kill_grp $SPID; fail "deadline: no time left for $L b$blk $c"; }
  grp $to python3 $I/e2e_client.py --i32 $R/tmp/p_$L.i32 --id $L --tag ${ARM}_${L}_$c --block $bn --out $out --port $SPORT \
      --server-pid $SPID --expect-model ${FILE[q38]} --expect-ctx ${CTX[$L]} --reps $REPS --n-predict $N_PREDICT $xd >> $R/e2e/client.log 2>&1
  rc=$?; kill_grp $SPID; wait $SPID 2>/dev/null
  echo "${L}_b${blk}_${c}_end $(date +%s)" >> $R/stages.txt
  # a row that ends past the deadline is not recorded (REV-032): its files are renamed and the run FAILS
  if past; then mv $out $out.late 2>/dev/null; mv $log $log.late 2>/dev/null
    fail "deadline: $L b$blk $c ended past the deadline (client rc $rc, timeout $to s), not recorded"; fi
  v=$(python3 $I/e2e_check.py srv $out --log $log --cfg $c --reps $REPS --n-predict $N_PREDICT --len $n ${SELREF[$L]:+--ref-sel ${SELREF[$L]}})
  case "$v" in
    OK*) ;;
    OOM*) [ $blk = 1 ] && OOMED[$L.$c]=1 ;;
    *) inc "E2E $L b$blk $c ${v:0:200} (client rc $rc)" ;;
  esac
  if past; then mv $out $out.late 2>/dev/null; mv $log $log.late 2>/dev/null
    fail "deadline: $L b$blk $c validated past the deadline, not recorded"; fi
  d=$(( $(date +%s) - t1 )); [ $d -gt ${EST[$L]} ] && EST[$L]=$d
  st "E2E $L b$blk $c ${v:0:300}"
}
mapfile -t MROWS < $R/matrix.txt
for row in "${MROWS[@]}"; do set -- $row; srv_case $1 $2 $3; done
kill $BG 2>/dev/null
finish
