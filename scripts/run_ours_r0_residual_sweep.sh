#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
DEMO_DATASET="${DEMO_DATASET:-outputs/robocerebra/ctr-recovery-v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-residual-v8-confirmed}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/Ours/search/R0-residual-v8-confirmed}"
COUNTERFACTUAL_PREFIX="${COUNTERFACTUAL_PREFIX:-ctr-counterfactual-rollout-v8-confirmed}"
RESIDUAL_OPTION_ADVANTAGES="${RESIDUAL_OPTION_ADVANTAGES:-false}"
MIN_STRICT_PREFERENCE_RATE="${MIN_STRICT_PREFERENCE_RATE:-0.10}"
RESIDUAL_STRATIFIED_SPLIT="${RESIDUAL_STRATIFIED_SPLIT:-false}"
COUNTERFACTUAL_DATASETS=(
  "outputs/robocerebra/${COUNTERFACTUAL_PREFIX}-seed10007"
  "outputs/robocerebra/${COUNTERFACTUAL_PREFIX}-seed11007"
  "outputs/robocerebra/${COUNTERFACTUAL_PREFIX}-seed12007"
)

mkdir -p "${CHECKPOINT_ROOT}" "${ARTIFACT_ROOT}"
additional_args=()
residual_advantage_args=()
split_args=()
if [[ "${RESIDUAL_OPTION_ADVANTAGES}" == "true" ]]; then
  residual_advantage_args=(--residual-option-advantages)
elif [[ "${RESIDUAL_OPTION_ADVANTAGES}" != "false" ]]; then
  echo "RESIDUAL_OPTION_ADVANTAGES must be true or false" >&2
  exit 1
fi
if [[ "${RESIDUAL_STRATIFIED_SPLIT}" == "true" ]]; then
  split_args=(--stratify-residual-overrides)
elif [[ "${RESIDUAL_STRATIFIED_SPLIT}" != "false" ]]; then
  echo "RESIDUAL_STRATIFIED_SPLIT must be true or false" >&2
  exit 1
fi
for dataset in "${COUNTERFACTUAL_DATASETS[@]}"; do
  PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_split \
    --corpus "${dataset}" \
    --modulus 5 \
    --remainder 4 \
    "${split_args[@]}" \
    >>"${ARTIFACT_ROOT}/option_split.log"
  PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_audit \
    --corpus "${dataset}" \
    --output "${ARTIFACT_ROOT}/$(basename "${dataset}")-audit.json" \
    --min-strict-preference-rate "${MIN_STRICT_PREFERENCE_RATE}" \
    >>"${ARTIFACT_ROOT}/corpus_audit.log"
  additional_args+=(--additional-train-dataset "${dataset}")
done

# id encoder history width layers heads feedforward dropout steps lr wd live-frac option-frac
# Frozen before any residual model or development-seed result is produced.
variants=(
  "v00-linear-h4-w32-d025 linear 4 32 1 4 128 0.25 1000 0.00010 0.10 0.50 0.25"
  "v01-linear-h8-w64-d025 linear 8 64 1 4 128 0.25 1500 0.00010 0.10 0.50 0.25"
  "v02-mlp-h4-w32-d050 mlp 4 32 1 4 128 0.50 1000 0.00010 0.10 0.50 0.25"
  "v03-mlp-h8-w64-d025 mlp 8 64 1 4 256 0.25 1500 0.00010 0.05 0.50 0.25"
  "v04-gru-h8-w64-d025 gru 8 64 1 4 256 0.25 1500 0.00010 0.05 0.50 0.25"
  "v05-transformer-h8-w64-d025 transformer 8 64 1 4 128 0.25 1500 0.00010 0.05 0.50 0.25"
)

for specification in "${variants[@]}"; do
  read -r variant encoder history width layers heads feedforward dropout steps lr wd live_fraction option_fraction <<<"${specification}"
  destination="${CHECKPOINT_ROOT}/${variant}"
  provenance="${destination}/ours_recovery_provenance.json"
  if [[ -f "${provenance}" ]]; then
    continue
  fi
  if [[ -d "${destination}" ]]; then
    echo "Refusing incomplete destination: ${destination}" >&2
    exit 1
  fi
  PYTHONPATH=src "${PYTHON_BIN}" -m unitree_gr00t.ours_train \
    --dataset "${DEMO_DATASET}" \
    "${additional_args[@]}" \
    --destination "${destination}" \
    --encoder "${encoder}" \
    --history-length "${history}" \
    --temporal-width "${width}" \
    --temporal-layers "${layers}" \
    --temporal-heads "${heads}" \
    --feedforward-width "${feedforward}" \
    --dropout "${dropout}" \
    --steps "${steps}" \
    --batch-size 128 \
    --learning-rate "${lr}" \
    --weight-decay "${wd}" \
    --warmup-steps 100 \
    --validate-steps 250 \
    --live-batch-fraction "${live_fraction}" \
    --option-batch-fraction "${option_fraction}" \
    --option-value-weight 0.10 \
    --option-rank-weight 0.25 \
    --option-classification-weight 1.00 \
    --checkpoint-selection residual \
    "${residual_advantage_args[@]}" \
    --device cuda:0 \
    --seed 10007 \
    >"${ARTIFACT_ROOT}/${variant}.log" 2>&1
done

PYTHONPATH=src "${PYTHON_BIN}" -m unitree_gr00t.ours_search \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --output "${ARTIFACT_ROOT}/completion_registry.json" \
  --expected-variants "${#variants[@]}"

PYTHONPATH=src "${PYTHON_BIN}" -m unitree_gr00t.ours_option_audit \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output "${ARTIFACT_ROOT}/residual_registry.json" \
  --device cuda:0 \
  --residual-retry-baseline \
  >"${ARTIFACT_ROOT}/residual_registry.log"
