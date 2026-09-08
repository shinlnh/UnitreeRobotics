#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
DEMO_DATASET="${DEMO_DATASET:-outputs/robocerebra/ctr-recovery-v1}"
LIVE_DATASET="${LIVE_DATASET:-outputs/robocerebra/ctr-live-corpus-v2/train-seed10007-full-c1-H16}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-live-v2}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/Ours/search/R0-live-v2}"

mkdir -p "${CHECKPOINT_ROOT}" "${ARTIFACT_ROOT}"

# encoder history width layers heads feedforward live_fraction
variants=(
  "v00-mlp-h16-live0125 mlp 16 128 1 4 512 0.125"
  "v01-mlp-h16-live0250 mlp 16 128 1 4 512 0.250"
  "v02-mlp-h16-live0500 mlp 16 128 1 4 512 0.500"
  "v03-mlp-h8-live0250 mlp 8 128 1 4 512 0.250"
  "v04-gru-h16-live0125 gru 16 128 1 4 512 0.125"
  "v05-gru-h16-live0250 gru 16 128 1 4 512 0.250"
  "v06-gru-h16-live0500 gru 16 128 1 4 512 0.500"
  "v07-transformer-h8-live0250 transformer 8 128 2 4 512 0.250"
)

for specification in "${variants[@]}"; do
  read -r variant encoder history width layers heads feedforward live_fraction <<<"${specification}"
  destination="${CHECKPOINT_ROOT}/${variant}"
  provenance="${destination}/ours_recovery_provenance.json"
  if [[ -f "${provenance}" ]]; then
    continue
  fi
  if [[ -d "${destination}" ]]; then
    echo "Refusing incomplete destination: ${destination}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m unitree_gr00t.ours_train \
    --dataset "${DEMO_DATASET}" \
    --additional-train-dataset "${LIVE_DATASET}" \
    --destination "${destination}" \
    --encoder "${encoder}" \
    --history-length "${history}" \
    --temporal-width "${width}" \
    --temporal-layers "${layers}" \
    --temporal-heads "${heads}" \
    --feedforward-width "${feedforward}" \
    --live-batch-fraction "${live_fraction}" \
    --steps 2000 \
    --batch-size 128 \
    --warmup-steps 100 \
    --validate-steps 250 \
    --device cuda:0 \
    --seed 10007 \
    >"${ARTIFACT_ROOT}/${variant}.log" 2>&1
done

"${PYTHON_BIN}" -m unitree_gr00t.ours_search \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --output "${ARTIFACT_ROOT}/registry.json" \
  --expected-variants "${#variants[@]}"
