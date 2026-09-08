#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${OURS_PYTHON:-${ROOT}/.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
DATASET="${ROOT}/outputs/robocerebra/ctr-recovery-v1"
CHECKPOINT_ROOT="${ROOT}/checkpoints/robocerebra/GR00T-RC-CTR-search/R0"
ARTIFACT_ROOT="${ROOT}/artifacts/Ours/search/R0"

mkdir -p "${CHECKPOINT_ROOT}" "${ARTIFACT_ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

VARIANTS=(
  "v00-linear-h4-w128-l1 linear 4 128 1"
  "v01-mlp-h4-w128-l1 mlp 4 128 1"
  "v02-mlp-h8-w128-l1 mlp 8 128 1"
  "v03-mlp-h16-w128-l1 mlp 16 128 1"
  "v04-gru-h4-w128-l1 gru 4 128 1"
  "v05-gru-h8-w128-l1 gru 8 128 1"
  "v06-gru-h16-w128-l1 gru 16 128 1"
  "v07-gru-h8-w128-l2 gru 8 128 2"
  "v08-transformer-h4-w128-l1 transformer 4 128 1"
  "v09-transformer-h8-w128-l1 transformer 8 128 1"
  "v10-transformer-h16-w128-l1 transformer 16 128 1"
  "v11-transformer-h8-w128-l2 transformer 8 128 2"
)

for specification in "${VARIANTS[@]}"; do
  read -r variant encoder history width layers <<<"${specification}"
  destination="${CHECKPOINT_ROOT}/${variant}"
  provenance="${destination}/ours_recovery_provenance.json"
  if [[ -f "${provenance}" ]]; then
    continue
  fi
  if [[ -d "${destination}" ]] && [[ -n "$(find "${destination}" -mindepth 1 -print -quit)" ]]; then
    echo "partial checkpoint blocks R0 resume: ${destination}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m unitree_gr00t.ours_train \
    --dataset "${DATASET}" \
    --destination "${destination}" \
    --encoder "${encoder}" \
    --history-length "${history}" \
    --temporal-width "${width}" \
    --temporal-layers "${layers}" \
    --temporal-heads 8 \
    --feedforward-width 512 \
    --steps 1500 \
    --batch-size 256 \
    --warmup-steps 100 \
    --validate-steps 250 \
    --device cuda:0 \
    --seed 10007 \
    >"${ARTIFACT_ROOT}/${variant}.log" 2>&1
done

"${PYTHON_BIN}" -m unitree_gr00t.ours_search \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --output "${ARTIFACT_ROOT}/registry.json" \
  --expected-variants 12
