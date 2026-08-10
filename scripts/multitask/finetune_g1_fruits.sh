#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"

GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
DATASET_ROOT="${GR00T_MULTITASK_DATASET_ROOT:-${PROJECT_ROOT}/datasets/g1_fruits_multitask_hf}"
BASE_MODEL="${GR00T_MULTITASK_BASE_MODEL:-${PROJECT_ROOT}/checkpoints/nvidia-gr00t-n1.7-3b}"
OUTPUT_DIR="${GR00T_MULTITASK_OUTPUT:-${PROJECT_ROOT}/checkpoints/groot-g1-fruits-multitask}"
: "${NUM_GPUS:?Set NUM_GPUS to the number of training GPUs}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
USE_WANDB="${USE_WANDB:-0}"
RESUME_TRAINING="${RESUME_TRAINING:-0}"

[[ -x "${GR00T_DIR}/.venv/bin/python" ]] || { echo "Run: make setup-groot" >&2; exit 1; }
[[ -f "${BASE_MODEL}/config.json" ]] || { echo "N1.7 base checkpoint is missing: ${BASE_MODEL}" >&2; exit 1; }

"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/multitask/audit_dataset.py" --root "${DATASET_ROOT}" --strict \
    --json-output "${PROJECT_ROOT}/outputs/g1_multitask_dataset_audit.json"
"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/multitask/validate_dataset.py" \
    --root "${DATASET_ROOT}"
"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/groot/check_model_access.py" \
    --repo "${GR00T_BACKBONE_MODEL}"

if [[ "${ALLOW_LOW_VRAM:-0}" != "1" ]] && command -v nvidia-smi >/dev/null; then
    mapfile -t selected_vram < <(
        nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n "${NUM_GPUS}"
    )
    if (( ${#selected_vram[@]} != NUM_GPUS )); then
        echo "Requested ${NUM_GPUS} GPUs, but detected ${#selected_vram[@]}." >&2
        exit 1
    fi
    min_vram_mib="$(printf '%s\n' "${selected_vram[@]}" | sort -n | head -n 1)"
    if (( min_vram_mib < 40000 )); then
        echo "GR00T N1.7 fine-tuning requires 40GB+ VRAM per GPU; smallest selected GPU has ${min_vram_mib} MiB." >&2
        echo "The local 16GB GPU is suitable for inference, not this official full fine-tune recipe." >&2
        exit 1
    fi
fi

dataset_paths=(
    "${DATASET_ROOT}/g1-pick-apple"
    "${DATASET_ROOT}/g1-pick-pear"
    "${DATASET_ROOT}/g1-pick-grapes"
    "${DATASET_ROOT}/g1-pick-starfruit"
)
dataset_soup="$(IFS=:; printf '%s' "${dataset_paths[*]}")"
extra_args=()
[[ "${USE_WANDB}" == "1" ]] && extra_args+=(--use-wandb)
[[ "${RESUME_TRAINING}" == "1" ]] && extra_args+=(--resume-from-checkpoint)

echo "Training ONE language-conditioned checkpoint from four G1 datasets:"
printf '  %s\n' "${dataset_paths[@]}"
cd "${GR00T_DIR}"
exec .venv/bin/python gr00t/experiment/launch_finetune.py \
    --base-model-path "${BASE_MODEL}" \
    --dataset-path "${dataset_soup}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${PROJECT_ROOT}/configs/groot/g1_fruits_multitask_config.py" \
    --experiment-name g1-fruits-multitask-n1.7 \
    --num-gpus "${NUM_GPUS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-total-limit 5 \
    --save-steps "${SAVE_STEPS}" \
    --max-steps "${MAX_STEPS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --ds-weights-alpha 0.0 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader-num-workers 4 \
    "${extra_args[@]}" \
    "$@"
