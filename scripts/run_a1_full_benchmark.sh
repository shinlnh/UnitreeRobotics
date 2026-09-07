#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ROBOCEREBRA_EXPERIMENT_ID="A1"
export ROBOCEREBRA_VARIANT="GR00T-RC"
export ROBOCEREBRA_CHECKPOINT="${PROJECT_ROOT}/checkpoints/robocerebra/GR00T-RC"
export ROBOCEREBRA_MODEL_REVISION="5d2e1e361bf65aabbe4d18179515f5a10936cc96"
export ROBOCEREBRA_ARTIFACT_ROOT="${PROJECT_ROOT}/artifacts/A1/full-benchmark"

exec "${PROJECT_ROOT}/scripts/run_a0_full_benchmark.sh"
