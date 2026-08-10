#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"

GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
DATASET_PATH="${GR00T_DATASET:-${PROJECT_ROOT}/datasets/lerobot/g1_navigation}"
OUTPUT_DIR="${GR00T_OUTPUT:-${PROJECT_ROOT}/checkpoints/groot-g1-navigation}"
LOCAL_BASE_MODEL="${PROJECT_ROOT}/checkpoints/nvidia-gr00t-n1.7-3b"
BASE_MODEL_PATH="${GR00T_BASE_MODEL:-${LOCAL_BASE_MODEL}}"

[[ -x "${GR00T_DIR}/.venv/bin/python" ]] || {
    echo "Run scripts/setup/bootstrap_groot.sh first." >&2
    exit 1
}
[[ -d "${DATASET_PATH}/meta" ]] || {
    echo "LeRobot dataset not found at ${DATASET_PATH}." >&2
    exit 1
}
if [[ ! -f "${BASE_MODEL_PATH}/config.json" ]]; then
    echo "Pinned local GR00T base model not found at ${BASE_MODEL_PATH}." >&2
    echo "Download revision ${GR00T_MODEL_REVISION} with:" >&2
    echo "  hf download ${GR00T_MODEL} --revision ${GR00T_MODEL_REVISION} --local-dir ${LOCAL_BASE_MODEL}" >&2
    exit 1
fi

# N1.7 constructs its visual-language backbone from this gated repository
# before restoring the GR00T post-training checkpoint. Fail before allocating
# the 3B model when the user has not accepted the model terms or authenticated.
"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/groot/check_model_access.py" \
    --repo "${GR00T_BACKBONE_MODEL}"

cd "${GR00T_DIR}"
exec uv run bash examples/finetune.sh \
    --base-model-path "${BASE_MODEL_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --modality-config-path "${PROJECT_ROOT}/configs/groot/g1_navigation_config.py" \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name g1-navigation \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --state-dropout-prob 0.1 \
    -- "$@"
