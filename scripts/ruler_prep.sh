#!/bin/bash
# Prepare RULER prompts (NVIDIA/RULER at a pinned commit), 20 samples per task, seed 42.
# Env: OUT (save dir), TOK (HF tokenizer id), MAX_LEN (default 131072), TEMPLATE (default base),
#      TASKS (default all 13), WORK (clone dir, default ./ext), PY (default python3).
# RULER's prepare.py calls `python` itself: PY's directory is put first on PATH, and it must provide `python`
# (a virtual environment does). Then tokenise: python ruler_tokens.py --data $OUT --model $TOK --out prompts.i32
set -euo pipefail
: "${OUT:?set OUT}" "${TOK:?set TOK}"
MAX_LEN=${MAX_LEN:-131072}; N=${N:-20}; TEMPLATE=${TEMPLATE:-base}; PY=$(command -v "${PY:-python3}")
RULER_COMMIT=${RULER_COMMIT:-db43f92ff634bbce1c33d1fb15858aec4bba509a}
TASKS=${TASKS:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2"}
mkdir -p "${WORK:-ext}" "$OUT"
WORK=$(cd "${WORK:-ext}" && pwd); OUT=$(cd "$OUT" && pwd)
export PATH="$(dirname "$PY"):$PATH"
command -v python > /dev/null || { echo "no 'python' on PATH next to $PY; use a virtual environment" >&2; exit 1; }
cd "$WORK"
[ -d RULER ] || git clone -q https://github.com/NVIDIA/RULER.git
git -C RULER checkout -q "$RULER_COMMIT"
git -C RULER rev-parse HEAD > "$OUT/ruler_commit.txt"
"$PY" -m pip install -q wonderwords nltk pyyaml pandas scipy tenacity html2text beautifulsoup4 tqdm transformers
"$PY" -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
cd RULER/scripts/data/synthetic/json
[ -f PaulGrahamEssays.json ] || "$PY" download_paulgraham_essay.py > "$OUT/essays.log" 2>&1
[ -f squad.json ] || bash download_qa_dataset.sh > "$OUT/qa_dl.log" 2>&1
cd "$WORK/RULER/scripts/data"
for t in $TASKS; do
  "$PY" prepare.py --save_dir "$OUT" --benchmark synthetic --task "$t" --tokenizer_path "$TOK" --tokenizer_type hf \
    --max_seq_length "$MAX_LEN" --num_samples "$N" --random_seed 42 --model_template_type "$TEMPLATE" > "$OUT/prep_$t.log" 2>&1
  n=$(wc -l < "$OUT/$t/validation.jsonl" 2> /dev/null || echo 0)
  [ "$n" -eq "$N" ] || { echo "$t: $n of $N samples, see $OUT/prep_$t.log" >&2; exit 1; }
done
echo "prepared $(echo $TASKS | wc -w) tasks x $N samples in $OUT"
