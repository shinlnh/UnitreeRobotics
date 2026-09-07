#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ROBOCEREBRA_EXPERIMENT_ID="B"
export ROBOCEREBRA_VARIANT="GR00T-RC-SparkVLA-style-execution"
export ROBOCEREBRA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/robocerebra/GR00T-RC"
export ROBOCEREBRA_MODEL_REVISION="5d2e1e361bf65aabbe4d18179515f5a10936cc96"
export ROBOCEREBRA_SELECTOR_CHECKPOINT="${PROJECT_ROOT}/checkpoints/robocerebra/GR00T-RC-SparkVLA-selector"
export ROBOCEREBRA_SELECTOR_METHOD="SparkVLA-unified-stop-prefix-GR00T-reimplementation"
export ROBOCEREBRA_SELECTOR_PAPER="arXiv:2608.16172v1"
export ROBOCEREBRA_STOP_CONFIRMATION_WINDOW="2"
export ROBOCEREBRA_SERVER_MODULE="unitree_gr00t.b_server"
export ROBOCEREBRA_EVALUATOR_MODULE="unitree_gr00t.b_eval"
export ROBOCEREBRA_MERGE_MODULE="unitree_gr00t.b_merge"
export ROBOCEREBRA_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/B/full-benchmark"
export ROBOCEREBRA_BASELINE_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/A2/full-benchmark"

exec "${PROJECT_ROOT}/scripts/run_a0_full_benchmark.sh"
