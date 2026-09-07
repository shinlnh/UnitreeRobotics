#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src" python3 -m unitree_gr00t.cli a2-check \
  --task-types Ideal Memory_Execution Memory_Exploration Mix Observation_Mismatching Random_Disturbance
exec "${PROJECT_ROOT}/scripts/run_a2_full_benchmark.sh"
