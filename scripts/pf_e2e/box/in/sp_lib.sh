# PORT-SP: functions shared by run_port_sp_a.sh and run_port_sp_b.sh (sourced, bash 4+).
# The caller sets W R I SRC B UP UB COMMIT BASE TOK, the FILE/URL/SHA arrays, CMK, and the timeouts below.
# Final status line: COMPLETE (every expected artifact present and validated), INCOMPLETE <reasons>, or FAILED <reason>.

SPARSE_VARS="LLAMA_SPARSE_DECODE LLAMA_SPARSE_DECODE_MIN_CTX LLAMA_SPARSE_DECODE_FUSED LLAMA_SPARSE_DECODE_SCORE LLAMA_SPARSE_DECODE_STATS"
for v in $SPARSE_VARS; do unset $v; done
# every process gets an explicit environment: dense runs clear the sparse controls, sparse runs set them after clearing
DENSE_ENV="env"; for v in $SPARSE_VARS; do DENSE_ENV="$DENSE_ENV -u $v"; done
SPARSE_ENV() { echo "$DENSE_ENV LLAMA_SPARSE_DECODE=$1 LLAMA_SPARSE_DECODE_MIN_CTX=${2:-0} LLAMA_SPARSE_DECODE_FUSED=3"; }
INC=""; BG=""; PGF=$R/pgids   # PGF: one process-group id per line, every heavy stage runs in its own session

st() { echo "$* $(date -Is)" >> $R/status.txt; }
inc() { INC="$INC; $*"; st "INCOMPLETE-ITEM $*"; }
fail() { st "FAILED $*"; exit 2; }
left() { [ $(date +%s) -lt $DEADLINE ]; }
m() { echo $W/${FILE[$1]}; }

# process groups: every heavy stage runs in its own session (setsid), so the whole group, reparented descendants included,
# is stopped with kill -- -PGID: TERM, then KILL after 20 s. The group ids are listed in $PGF; the wrapper kills any left.
kill_grp() {
  local g=$1 i
  kill -0 -- -$g 2>/dev/null || return 0
  kill -TERM -- -$g 2>/dev/null
  for i in $(seq 20); do kill -0 -- -$g 2>/dev/null || return 0; sleep 1; done
  kill -KILL -- -$g 2>/dev/null; return 0
}
grp() {  # <timeout s> cmd...: run in a new session with a stage timeout, wait for it; redirections apply to the command
  local t=$1 p rc; shift
  setsid -w timeout -k 30 $t "$@" &
  p=$!; echo $p >> $PGF
  wait $p; rc=$?
  kill_grp $p
  return $rc
}
grp_bg() { setsid "$@" & GRP_PID=$!; echo $GRP_PID >> $PGF; }   # background stage in its own group
cleanup() { local p g; for p in $BG; do kill $p 2>/dev/null; done; [ -f $PGF ] && for g in $(cat $PGF); do kill_grp $g; done; }
trap cleanup EXIT
trap 'st "INCOMPLETE killed by TERM (hard cap or operator)"; exit 143' TERM

dl() { [ -e $(m $1) ] || { grp 7200 aria2c -q -x16 -s16 -d $W -o ${FILE[$1]}.part ${URL[$1]}/${FILE[$1]} && mv $(m $1).part $(m $1); } > $R/dl_$1.log 2>&1; }
shaok() { local s; s=$(sha256sum $(m $1) | cut -d' ' -f1); echo "$s ${FILE[$1]}" >> $R/sha.txt; [ "$s" = "${SHA[$1]}" ]; }

# ---- builds: clean tree at the pinned commit, fresh build directory, effective CMake settings recorded for comparison
fetch_fork() {
  [ -d $SRC/.git ] || git clone -q https://github.com/topppg/llama.cpp.git $SRC || fail "clone"
  git -C $SRC fetch -q origin sparse-decode-upstream-20261117 && git -C $SRC checkout -q --detach $COMMIT || fail "checkout"
  git -C $SRC rev-parse HEAD > $R/commit.txt
  grep -q "^$COMMIT$" $R/commit.txt || fail "commit $(cat $R/commit.txt)"
  [ -z "$(git -C $SRC status --porcelain --untracked-files=no)" ] || fail "fork tree has tracked changes"
}
fetch_upstream() {
  git -C $SRC merge-base --is-ancestor $BASE $COMMIT || fail "$BASE is not the fork's base"
  [ -d $UP ] || git -C $SRC worktree add -q --detach $UP $BASE || fail "worktree"
  git -C $UP rev-parse HEAD > $R/commit_upstream.txt
  grep -q "^$BASE$" $R/commit_upstream.txt && [ -z "$(git -C $UP status --porcelain --untracked-files=no)" ] || fail "upstream tree"
}
build() {  # src builddir tag targets... (returns non-zero on failure; the caller decides)
  local s=$1 b=$2 tag=$3; shift 3
  rm -rf $b
  grp 3600 cmake -B $b -S $s $CMK > $R/cmake_$tag.log 2>&1 || return 1
  grp 5400 cmake --build $b -j $(nproc) --target "$@" > $R/build_$tag.log 2>&1 || return 1
  grep -E "^(CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE|GGML_CUDA|GGML_NATIVE|GGML_CUDA_FA_ALL_QUANTS|CMAKE_CUDA_COMPILER):" \
      $b/CMakeCache.txt 2>/dev/null | sort > $R/cmake_$tag.settings
}

# ---- gates
gate_k() {
  local op line n mm
  for op in PAGE_TOP_K FLASH_ATTN_EXT; do
    grp 3600 $B/test-backend-ops test -b CUDA0 -o $op > $R/tbo_$op.txt 2>&1 || fail "gate K $op rc=$?"
    line=$(grep -oE '[0-9]+/[0-9]+ tests passed' $R/tbo_$op.txt | tail -1)
    n=${line%%/*}; mm=${line#*/}; mm=${mm%% *}
    [ -n "$line" ] && [ "$n" -gt 0 ] && [ "$n" = "$mm" ] && ! grep -q "FAIL" $R/tbo_$op.txt || fail "gate K $op: '${line:-no test count}'"
    st "GATE K PASS $op $line"
  done
}
gate_a() {  # model: with a budget above the prompt every page is selected, so greedy outputs must equal dense
  local k=$1
  rm -f $W/a.i32
  python3 $I/ruler_subset.py $TOK $W/a.i32 --per-task 1 --tasks 0 > /dev/null || fail "gate A subset"
  grp 1800 $DENSE_ENV $B/llama-sparse-tail -m $(m $k) --kv-type q8_0 --gen $W/a.i32 --configs dense,quest:262144:f3 \
      > $R/ident/$k.jsonl 2> $R/ident/$k.err || fail "gate A $k run rc=$?"
  python3 - $R/ident/$k.jsonl <<'EOF' || fail "gate A $k outputs differ or are malformed"
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok = (sorted(x["config"] for x in r) == ["dense", "quest:262144:f3"] and {x["id"] for x in r} == {0}
      and all(len(x["out"]) > 0 for x in r) and r[0]["out"] == r[1]["out"])
sys.exit(0 if ok else 1)
EOF
  st "GATE A PASS $k"
}

# ---- speed: the static matrix file ($MATRIX, part of the pre-registration) lists every row in run order:
# model build kv cfg block depth. One depth per llama-bench call, unique file per row, outcome OK / OOM / FAIL.
lb1() {  # <model> <build fork|up> <kv> <cfg dense|fa0|budget> <block> <depth>
  local k=$1 build=$2 kv=$3 cfg=$4 blk=$5 d=$6 fa=1 envs="$DENSE_ENV" rc cls bin=$B
  local tag=${build}_${kv}_${cfg} f=$R/bench/${k}_${build}_${kv}_${cfg}_b${blk}_d${d}
  [ -e $f.jsonl ] && fail "duplicate bench $f"
  [ $build = up ] && bin=$UB
  [ $cfg = fa0 ] && fa=0
  case $cfg in dense|fa0) ;; *) envs=$(SPARSE_ENV $cfg 0);; esac
  grp $BENCH_TO $envs $bin/llama-bench -m $(m $k) -ngl 99 -fa $fa -p 0 -n 64 -r $REPS -d $d -ctk $kv -ctv $kv -o jsonl \
      > $f.jsonl 2> $f.err
  rc=$?; cls=OK
  if [ $rc != 0 ]; then
    grep -qiE "out of memory|cudaMalloc failed|failed to allocate|unable to allocate" $f.err && cls=OOM || cls=FAIL
  elif [ "$(grep -c '^{' $f.jsonl)" != 1 ]; then
    cls=FAIL
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' $k $tag $blk $d $rc $cls >> $R/bench/outcomes.tsv
  [ $cls = OK ] || st "BENCH $k $tag b$blk d=$d $cls rc=$rc"
  [ $cls = FAIL ] && inc "bench $k $tag b$blk d=$d failed (rc=$rc)"
  [ $cls = OK ]
}
run_matrix() {  # model: every row of the static matrix for that model, in file order
  local k=$1 mk bd kv cfg blk d n=0
  while read -r mk bd kv cfg blk d; do
    case "$mk" in ""|"#"*) continue;; esac
    [ "$mk" = "$k" ] || continue
    lb1 $mk $bd $kv $cfg $blk $d; n=$((n+1))
  done < <(cat $MATRIX)
  [ $n -gt 0 ] || inc "matrix has no rows for $k"
}
outcome() { awk -F'\t' -v k=$1 -v t=$2 -v b=$3 -v d=$4 '$1==k&&$2==t&&$3==b&&$4==d{print $6}' $R/bench/outcomes.tsv | tail -1; }
audit_line() {  # model bw: nominal bytes, tables, matrix check
  local k=$1 bw=$2 kv
  for kv in q8_0 f16; do
    python3 $I/sp_bytes.py $(m $k) --kv $kv --depths 0,32768,65536,98304,131072 --bw $bw --json $R/audit/bytes_${k}_$kv.json > /dev/null
  done
  python3 $I/sp_audit.py bench $R/bench $k --matrix $MATRIX --bytes q8_0=$R/audit/bytes_${k}_q8_0.json,f16=$R/audit/bytes_${k}_f16.json \
      --bw $bw --out $R/audit/audit_$k.json --line > $R/audit/line_$k.txt 2> $R/audit/audit_$k.err || inc "audit $k: bench matrix incomplete"
  st "AUDIT $k $(head -c 1500 $R/audit/line_$k.txt)"
}

# ---- RULER: one file per (model, pass, task); pass 1 = prompts 0-9, pass 2 = prompts 10-19; configs dense, quest:27648:f3
ruler_task() {
  local k=$1 p=$2 t=$3 sel f=$R/ruler/${1}_p${2}_t${3}
  [ -e $f.jsonl ] && fail "duplicate ruler $f"
  rm -f $W/r.i32
  if [ $p = 1 ]; then sel="--per-task 10 --tasks $t"
  else sel="--per-task 20 --tasks $t --ids $(seq $((t*1000+10)) $((t*1000+19)) | paste -sd, -)"; fi
  python3 $I/ruler_subset.py $TOK $W/r.i32 $sel > /dev/null || { inc "ruler subset $k p$p t$t"; return 1; }
  grp $RULER_TO $DENSE_ENV $B/llama-sparse-tail -m $(m $k) --kv-type q8_0 --gen $W/r.i32 --configs dense,quest:27648:f3 \
      > $f.jsonl 2> $f.err || { inc "ruler $k p$p t$t rc=$?"; return 1; }
  st "RULER $k pass $p task $t done"
}
gate_d_wait() {  # model: the run continues only after the Mac's verdict on pass 1 task 0 (scored there); no verdict = STOP
  local k=$1 t_end=$(( T_TASK0 + GATE_D_TIMEOUT ))
  st "GATE D WAIT $k: touch $R/GATE_D_PASS_$k or $R/STOP_RULER; no verdict by $(date -Is -d @$t_end) = STOP"
  while :; do
    [ -e $R/STOP_RULER ] && { st "GATE D FAIL $k (STOP_RULER)"; inc "gate D failed on $k: RULER stopped"; return 1; }
    [ -e $R/GATE_D_PASS_$k ] && { st "GATE D PASS $k (acknowledged)"; return 0; }
    [ $(date +%s) -ge $t_end ] && { st "GATE D NO VERDICT $k by timeout"; inc "gate D not acknowledged on $k: RULER stopped"; return 1; }
    sleep 20
  done
}
ruler_rest() {  # model: pass 1 tasks 1-12, then pass 2 tasks 0-12, each started only before the deadline
  local k=$1 p t
  for p in 1 2; do
    for t in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
      [ $p = 1 ] && [ $t = 0 ] && continue
      [ -e $R/STOP_RULER ] && { inc "RULER $k stopped by STOP_RULER before pass $p task $t"; return 1; }
      left || { inc "RULER $k deadline before pass $p task $t"; return 1; }
      ruler_task $k $p $t || return 1
    done
  done
}
ruler_check() {  # model: exact coverage, no duplicates; complete pairs written for scoring
  python3 $I/sp_check.py ruler $R/ruler $1 --out $R/ruler/${1}_pairs.jsonl > $R/ruler/${1}_check.json 2>&1 ||
      inc "RULER $1 coverage incomplete ($(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('n_complete_pairs'), 'of', d.get('n_expected_pairs'))" $R/ruler/${1}_check.json 2>/dev/null))"
}

# ---- end to end through llama-server: a fresh server on a random port per (config, block); identity checked on /props
srv_start() {  # tag kv sparse mtp model [draft model]
  local tag=$1 kv=$2 sp=$3 mtp=$4 k=$5 dk=${6:-} extra="" envs="$DENSE_ENV"
  if [ $mtp = 1 ]; then
    extra="--spec-type draft-mtp -ctkd $kv -ctvd $kv"
    [ -n "$dk" ] && extra="$extra -md $(m $dk) -ngld 99"
  fi
  [ $sp = 1 ] && envs=$(SPARSE_ENV 27648 32768)
  SPORT=$(( 20000 + RANDOM % 20000 ))
  setsid $envs $B/llama-server -m $(m $k) -ngl 99 -fa on -c 132096 -np 1 -b 2048 -ub 512 -ctk $kv -ctv $kv --cache-ram 0 \
      --host 127.0.0.1 --port $SPORT $extra > $R/e2e/srv_${k}_${tag}_$SPORT.log 2>&1 &
  SPID=$!; echo $SPID >> $PGF   # its own session: SPID is the group id
}
e2e_cfg() {  # tag kv sparse mtp model draft block out mode(timed|smoke)
  local tag=$1 kv=$2 sp=$3 mtp=$4 k=$5 dk=$6 blk=$7 out=$8 mode=$9 xd="" rc
  srv_start $tag $kv $sp $mtp $k $dk
  [ $mtp = 1 ] && xd="--expect-draft"
  if [ $mode = smoke ]; then
    grp 900 python3 $I/sp_e2e.py run --i32 $TOK --id $E2E_ID --tag $tag --block 0 --out $out --port $SPORT --server-pid $SPID \
        --expect-model ${FILE[$k]} --expect-ctx 132096 --smoke $xd >> $R/e2e/client_$k.log 2>&1
  else
    grp $E2E_TO python3 $I/sp_e2e.py run --i32 $TOK --id $E2E_ID --tag $tag --block $blk --out $out --port $SPORT --server-pid $SPID \
        --expect-model ${FILE[$k]} --expect-ctx 132096 --warmup --reps 2 $xd >> $R/e2e/client_$k.log 2>&1
  fi
  rc=$?
  kill_grp $SPID; wait $SPID 2>/dev/null
  [ $rc = 0 ] || st "E2E $k $tag block $blk rc=$rc"
  return $rc
}
e2e_run() {  # model draft "tag kv sparse mtp;..." : block 1 in the listed order, block 2 reversed; then the strict summary
  local k=$1 dk=$2 specs=$3 out=$R/e2e/e2e_$1.jsonl s tags="" list=() i
  IFS=';' read -ra list <<< "$specs"
  for s in "${list[@]}"; do set -- $s; tags="$tags,$1"; e2e_cfg $1 $2 $3 $4 $k "$dk" 1 $out timed; done
  for (( i=${#list[@]}-1; i>=0; i-- )); do set -- ${list[$i]}; e2e_cfg $1 $2 $3 $4 $k "$dk" 2 $out timed; done
  python3 $I/sp_e2e.py summary $out --expect-tags ${tags#,} --reps 4 > $R/e2e/summary_$k.json 2>&1 || inc "E2E $k incomplete or invalid"
  st "E2E $k summary $(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('status'))" $R/e2e/summary_$k.json 2>/dev/null)"
}

finish() {
  if [ -z "$INC" ]; then st "COMPLETE"; exit 0; fi
  st "INCOMPLETE${INC}"; exit 3
}
