#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI=(/usr/bin/env "PYTHONPATH=${PROJECT_ROOT}/src" python3 -m unitree_gr00t.cli)

cd "${PROJECT_ROOT}"
"${CLI[@]}" b-check --stage checkpoint
"${CLI[@]}" b-retry-check
exec "${PROJECT_ROOT}/scripts/run_b_retry_full_benchmark.sh"
