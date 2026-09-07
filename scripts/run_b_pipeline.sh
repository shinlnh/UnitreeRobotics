#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI=(/usr/bin/env "PYTHONPATH=${PROJECT_ROOT}/src" python3 -m unitree_gr00t.cli)

cd "${PROJECT_ROOT}"
if [[ ! -f outputs/robocerebra/sparkvla-selector-v1/manifest.json ]]; then
  "${CLI[@]}" b-prepare --execute
fi
"${CLI[@]}" b-check --stage index
if ! "${CLI[@]}" b-check --stage features >/dev/null 2>&1; then
  "${CLI[@]}" b-features --batch-size 64 --execute
fi
"${CLI[@]}" b-check --stage features
if [[ ! -f checkpoints/robocerebra/GR00T-RC-SparkVLA-selector/model.safetensors ]]; then
  "${CLI[@]}" b-train --execute
fi
"${CLI[@]}" b-check --stage checkpoint
exec "${PROJECT_ROOT}/scripts/run_b_full_benchmark.sh"
