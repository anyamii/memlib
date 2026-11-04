#!/usr/bin/env bash
# LoCoMo CC BY-NC 4.0: review upstream terms before running.
# https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/LICENSE.txt
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: scripts/download_locomo.sh OUTSIDE_REPO/locomo10.json' >&2
  exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
target="$(python3 - "$1" "$repo" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]).resolve()
if path.is_relative_to(Path(sys.argv[2])):
    raise SystemExit('download destination must be outside the repository')
if path.exists():
    raise SystemExit('refusing to overwrite existing destination')
print(path)
PY
)"
mkdir -p "$(dirname "$target")"
temp="$(mktemp "${target}.XXXXXX")"
trap 'rm -f "$temp"' EXIT
curl -fL 'https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json' -o "$temp"
printf '%s  %s\n' '79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4' "$temp" | sha256sum -c -
mv -n "$temp" "$target"
