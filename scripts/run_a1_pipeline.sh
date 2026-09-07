#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"
A1_CONVERSION_WORKERS="${A1_CONVERSION_WORKERS:-10}" \
  "${PROJECT_ROOT}/scripts/run_a1_prepare_dataset.sh"
PYTHONPATH="${PROJECT_ROOT}/src" "${PROJECT_ROOT}/.venv/bin/python" \
  -m unitree_gr00t.cli a1-train --execute
"${PROJECT_ROOT}/scripts/run_a1_full_benchmark.sh"
