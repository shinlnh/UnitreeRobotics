#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
GR00T_ROOT="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
RAW_DIR="${1:-${PROJECT_ROOT}/datasets/raw/g1_navigation}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/datasets/lerobot/g1_navigation}"
FPS="${FPS:-10}"

if [[ ! -x "${GR00T_ROOT}/.venv/bin/python" ]]; then
  echo "GR00T environment missing. Run: bash scripts/setup/bootstrap_groot.sh" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
exec "${GR00T_ROOT}/.venv/bin/python" tools/lerobot/convert_raw_navigation.py \
  --raw-dir "${RAW_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --fps "${FPS}"
