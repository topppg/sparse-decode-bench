#!/bin/bash
# Rerun the PF-E2E decision (e2e_decide.py, with and without --amend4) and the PF-NT decision (pfn_decide.py) on the
# committed results, unchanged scripts, no GPU.
# The decision scripts find their helpers and inputs through the project repository's layout (experiments/src,
# experiments/run_scripts/<protocol>, experiments/results/<tag>). This wrapper rebuilds that layout in a temporary
# directory: scripts and result folders copied, *.err.gz decompressed (the llama.cpp logs are committed gzipped), and runs
# each script from experiments/src with the relative arguments of the committed run, so its JSON can be compared byte for
# byte with the committed file.
# Usage: scripts/pf_e2e/decide.sh OUT_DIR
#   writes OUT_DIR/{decision.json, decision_as_registered.json, pf_nt_decision.json}
#   PY: the Python command (default: uv with transformers and numpy). Scoring downloads the Qwen/Qwen3.8-27B tokenizer at
#   revision 37561816e66b5b9153b1b64c10968c18c1e6f4a0 from Hugging Face unless it is cached.
set -eu
O=$(mkdir -p "$1" && cd "$1" && pwd)
B=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-uv run -q --no-project --python 3.12 --with transformers --with numpy python}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
X=$T/experiments; R=$X/results
mkdir -p $X/src $X/run_scripts $R/pf_e2e/analysis $R/pf_e2e/inputs $R/pf_nt/analysis $R/pf_nt/inputs $R/pf_nt/vastai38
cp $B/scripts/pf_e2e/src/port_r_score.py $X/src/
cp -R $B/scripts/pf_e2e $X/run_scripts/pf_e2e
cp -R $B/scripts/pf_nt $X/run_scripts/pf_nt
mkdir -p $R/pf_e2e/vastai48 && cp -R $B/results/cuda_5090_pfe2e $R/pf_e2e/vastai48/20261121T222603Z
mkdir -p $R/pf_e2e/vastai51 && cp -R $B/results/cuda_4090_pfe2e $R/pf_e2e/vastai51/20261121T225917Z
mkdir -p $R/pf_e2e/vastai49 && cp -R $B/results/cuda_5090_pfe2e_ruler/vastai49 $R/pf_e2e/vastai49/20261121T222607Z
mkdir -p $R/pf_e2e/vastai50 && cp -R $B/results/cuda_5090_pfe2e_ruler/vastai50 $R/pf_e2e/vastai50/20261121T222610Z
cp $B/results/pfe2e_analysis/manifest.json $R/pf_e2e/analysis/
cp $B/results/pfe2e_analysis/inputs/*.meta.json $R/pf_e2e/inputs/
cp $B/results/cuda_6000_pfnt/analysis/manifest.json $B/results/cuda_6000_pfnt/analysis/decision.json $R/pf_nt/analysis/
cp $B/results/cuda_6000_pfnt/inputs/pf_nt_q38.i32.meta.json $R/pf_nt/inputs/
cp -R $B/results/cuda_6000_pfnt/vastai38/20261119T103525Z $B/results/cuda_6000_pfnt/vastai38/20261119T190800Z $R/pf_nt/vastai38/
find $R -name '*.err.gz' -exec gunzip {} +
cd $X/src
$PY ../run_scripts/pf_e2e/e2e_decide.py --speed-a ../results/pf_e2e/vastai48/20261121T222603Z \
  --speed-d ../results/pf_e2e/vastai51/20261121T225917Z --ruler-manifest ../results/pf_e2e/analysis/manifest.json \
  --amend4 --out $O/decision.json
$PY ../run_scripts/pf_e2e/e2e_decide.py --speed-a ../results/pf_e2e/vastai48/20261121T222603Z \
  --speed-d ../results/pf_e2e/vastai51/20261121T225917Z --ruler-manifest ../results/pf_e2e/analysis/manifest.json \
  --out $O/decision_as_registered.json
# PF-NT: the raw-prompt runs it compares with descriptively (PF-BF16, PF-CONF) are not published here, so its vs_raw
# fields come out as missing; the paths are the committed run's
$PY ../run_scripts/pf_nt/pfn_decide.py --manifest ../results/pf_nt/analysis/manifest.json --late F --late D \
  --raw-b38 ../results/pf_bf16/analysis/decision.json --raw-q38 ../results/pf_conf/analysis/decision_q38.json \
  --out $O/pf_nt_decision.json
