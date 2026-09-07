#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ROBOCEREBRA_EXPERIMENT_ID="A2"
export ROBOCEREBRA_VARIANT="GR00T-RC-fixed-hierarchy"
export ROBOCEREBRA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/robocerebra/GR00T-RC"
export ROBOCEREBRA_MODEL_REVISION="5d2e1e361bf65aabbe4d18179515f5a10936cc96"
export ROBOCEREBRA_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/A2/full-benchmark"
export ROBOCEREBRA_EVALUATOR_MODULE="unitree_gr00t.a2_eval"
export ROBOCEREBRA_MERGE_MODULE="unitree_gr00t.a2_merge"
export ROBOCEREBRA_BASELINE_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/A1/full-benchmark"

exec "${PROJECT_ROOT}/scripts/run_a0_full_benchmark.sh"
