#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"

DATASET_PATH="${SONIC_FETCH_DATASET:-${PROJECT_ROOT}/datasets/sonic/g1_fetch_clean}"
GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
HF=(uv tool run --from "huggingface_hub==${HF_HUB_CLI_VERSION}" hf)

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -x "${GR00T_DIR}/.venv/bin/python" ]] || {
    echo "GR00T environment is incomplete. Run: make setup-groot" >&2
    exit 1
}

"${HF[@]}" download "${GR00T_FETCH_DATASET}" \
    --repo-type dataset \
    --revision "${GR00T_FETCH_DATASET_REVISION}" \
    --local-dir "${DATASET_PATH}" \
    --max-workers "${HF_DOWNLOAD_WORKERS:-2}"
"${HF[@]}" cache verify "${GR00T_FETCH_DATASET}" \
    --repo-type dataset \
    --revision "${GR00T_FETCH_DATASET_REVISION}" \
    --local-dir "${DATASET_PATH}" \
    --fail-on-missing-files

"${GR00T_DIR}/.venv/bin/python" "${GR00T_DIR}/gr00t/data/stats.py" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${DATASET_PATH}/new_embodiment_config_defaults.py"

"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/sonic/audit_fetch_dataset.py" \
    --dataset "${DATASET_PATH}" --strict
"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/sonic/validate_fetch_dataset.py" \
    --dataset "${DATASET_PATH}"
