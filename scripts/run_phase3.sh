#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
artifacts="$(dirname "$PWD")/memlib-artifacts"
out="$artifacts/results-phase3"
for ((i=1; i<=$#; i++)); do
  if [[ "${!i}" == --out ]]; then j=$((i+1)); out="${!j}"; fi
done
mkdir -p "$out"
exec 9>"$out/.run.lock"
flock -n 9 || { echo 'Another run owns this output directory'; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -u -m eval.phase3 --data "$artifacts/locomo10.json" --cache "$artifacts/models" --out "$out" "$@" 2>&1 | tee -a "$out/run.log"
