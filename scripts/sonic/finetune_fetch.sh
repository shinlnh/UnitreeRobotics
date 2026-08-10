#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"

GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
DATASET_PATH="${SONIC_FETCH_DATASET:-${PROJECT_ROOT}/datasets/sonic/g1_fetch_clean}"
OUTPUT_DIR="${GR00T_FETCH_OUTPUT:-${PROJECT_ROOT}/checkpoints/groot-g1-sonic-fetch}"
: "${NUM_GPUS:?Set NUM_GPUS to the number of training GPUs}"
MAX_STEPS="${MAX_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
USE_WANDB="${USE_WANDB:-0}"

[[ -x "${GR00T_DIR}/.venv/bin/python" ]] || {
    echo "GR00T environment is incomplete. Run: make setup-groot" >&2
    exit 1
}
[[ -f "${DATASET_PATH}/meta/modality.json" ]] || {
    echo "Clean SONIC fetch dataset not found: ${DATASET_PATH}" >&2
    exit 1
}

"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/groot/check_model_access.py" \
    --repo "${GR00T_BACKBONE_MODEL}"

"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/sonic/audit_fetch_dataset.py" \
    --dataset "${DATASET_PATH}" --strict
"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/sonic/validate_fetch_dataset.py" \
    --dataset "${DATASET_PATH}"

if [[ -f "${DATASET_PATH}/new_embodiment_config_defaults.py" ]]; then
    FETCH_EMBODIMENT_TAG="${FETCH_EMBODIMENT_TAG:-NEW_EMBODIMENT}"
    FETCH_MODALITY_CONFIG="${FETCH_MODALITY_CONFIG:-${DATASET_PATH}/new_embodiment_config_defaults.py}"
else
    FETCH_EMBODIMENT_TAG="${FETCH_EMBODIMENT_TAG:-UNITREE_G1_SONIC}"
    FETCH_MODALITY_CONFIG="${FETCH_MODALITY_CONFIG:-${GR00T_DIR}/gr00t/configs/data/embodiment_configs.py}"
fi

if [[ "${ALLOW_LOW_VRAM:-0}" != "1" ]] && command -v nvidia-smi >/dev/null; then
    mapfile -t selected_vram < <(
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n "${NUM_GPUS}"
    )
    if (( ${#selected_vram[@]} != NUM_GPUS )); then
        echo "Requested ${NUM_GPUS} GPUs, but only ${#selected_vram[@]} were detected." >&2
        exit 1
    fi
    min_vram_mib="$(printf '%s\n' "${selected_vram[@]}" | sort -n | head -n 1)"
    if (( min_vram_mib < 40000 )); then
        echo "Smallest selected GPU has ${min_vram_mib} MiB; standard N1.7 fine-tuning requires 40GB+ per GPU." >&2
        echo "Use 40GB+ GPUs, or set ALLOW_LOW_VRAM=1 only to test an experimental recipe." >&2
        exit 1
    fi
fi

wandb_args=()
if [[ "${USE_WANDB}" == "1" ]]; then
    wandb_args+=(--use-wandb)
fi

cd "${GR00T_DIR}"
exec .venv/bin/python gr00t/experiment/launch_finetune.py \
    --base-model-path "${GR00T_MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag "${FETCH_EMBODIMENT_TAG}" \
    --modality-config-path "${FETCH_MODALITY_CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps "${SAVE_STEPS}" \
    --max-steps "${MAX_STEPS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    "${wandb_args[@]}" \
    "$@"
