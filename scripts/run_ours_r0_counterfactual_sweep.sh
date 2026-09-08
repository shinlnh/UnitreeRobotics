#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
DEMO_DATASET="${DEMO_DATASET:-outputs/robocerebra/ctr-recovery-v1}"
COUNTERFACTUAL_DATASET="${COUNTERFACTUAL_DATASET:-outputs/robocerebra/ctr-counterfactual-rollout-v3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-counterfactual-v3}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/Ours/search/R0-counterfactual-v3}"

mkdir -p "${CHECKPOINT_ROOT}" "${ARTIFACT_ROOT}"
PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_split \
  --corpus "${COUNTERFACTUAL_DATASET}" \
  --modulus 5 \
  --remainder 4 \
  >"${ARTIFACT_ROOT}/option_split.log"
PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_audit \
  --corpus "${COUNTERFACTUAL_DATASET}" \
  --output "${ARTIFACT_ROOT}/corpus_audit.json" \
  >"${ARTIFACT_ROOT}/corpus_audit.log"

# encoder history width layers heads feedforward live-fraction option-fraction
variants=(
  "v00-mlp-h16-live025-opt0125 mlp 16 128 1 4 512 0.25 0.125"
  "v01-mlp-h16-live050-opt0250 mlp 16 128 1 4 512 0.50 0.250"
  "v02-gru-h16-live050-opt0250 gru 16 128 1 4 512 0.50 0.250"
  "v03-transformer-h8-live025-opt0125 transformer 8 128 2 4 512 0.25 0.125"
)

for specification in "${variants[@]}"; do
  read -r variant encoder history width layers heads feedforward live_fraction option_fraction <<<"${specification}"
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
    --additional-train-dataset "${COUNTERFACTUAL_DATASET}" \
    --destination "${destination}" \
    --encoder "${encoder}" \
    --history-length "${history}" \
    --temporal-width "${width}" \
    --temporal-layers "${layers}" \
    --temporal-heads "${heads}" \
    --feedforward-width "${feedforward}" \
    --live-batch-fraction "${live_fraction}" \
    --option-batch-fraction "${option_fraction}" \
    --steps 2000 \
    --batch-size 128 \
    --warmup-steps 100 \
    --validate-steps 250 \
    --device cuda:0 \
    --seed 10007 \
    >"${ARTIFACT_ROOT}/${variant}.log" 2>&1
done

PYTHONPATH=src "${PYTHON_BIN}" -m unitree_gr00t.ours_search \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --output "${ARTIFACT_ROOT}/registry.json" \
  --expected-variants "${#variants[@]}"

PYTHONPATH=src "${PYTHON_BIN}" -m unitree_gr00t.ours_option_audit \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --output "${ARTIFACT_ROOT}/option_fit_audit.json" \
  --device cuda:0 \
  >"${ARTIFACT_ROOT}/option_fit_audit.log"
