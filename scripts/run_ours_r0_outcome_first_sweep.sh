#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export COUNTERFACTUAL_PREFIX="ctr-counterfactual-rollout-v9-outcome-first"
export CHECKPOINT_ROOT="checkpoints/robocerebra/GR00T-RC-CTR-search/R0-residual-v9-outcome-first"
export ARTIFACT_ROOT="artifacts/Ours/search/R0-residual-v9-outcome-first"
export RESIDUAL_OPTION_ADVANTAGES="true"
export RESIDUAL_STRATIFIED_SPLIT="true"
export MIN_STRICT_PREFERENCE_RATE="0.05"

exec "${ROOT}/scripts/run_ours_r0_residual_sweep.sh"
