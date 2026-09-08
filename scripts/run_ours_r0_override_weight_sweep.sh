#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Frozen after the unit-weight binary ablation and before any model below is
# trained. All four weights use the same six registered low-capacity models.
weights=(
  "w005 0.05"
  "w010 0.10"
  "w025 0.25"
  "w050 0.50"
)

for specification in "${weights[@]}"; do
  read -r weight_id weight <<<"${specification}"
  COUNTERFACTUAL_PREFIX="ctr-counterfactual-rollout-v9-outcome-first" \
  CHECKPOINT_ROOT="checkpoints/robocerebra/GR00T-RC-CTR-search/R0-residual-v9-override-weight/${weight_id}" \
  ARTIFACT_ROOT="artifacts/Ours/search/R0-residual-v9-override-weight/${weight_id}" \
  RESIDUAL_OPTION_ADVANTAGES="true" \
  RESIDUAL_STRATIFIED_SPLIT="true" \
  OPTION_OVERRIDE_WEIGHT="${weight}" \
  MIN_STRICT_PREFERENCE_RATE="0.05" \
    scripts/run_ours_r0_residual_sweep.sh
done
