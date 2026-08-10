#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

DATASET_ROOT="${GR00T_MULTITASK_DATASET_ROOT:-${PROJECT_ROOT}/datasets/g1_fruits_multitask_hf}"
DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-2}"
MAX_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-8}"

command -v hf >/dev/null || {
    echo "hf CLI is required. Install it with: uv tool install 'huggingface_hub==${HF_HUB_CLI_VERSION}'" >&2
    exit 1
}
if ! [[ "${DOWNLOAD_WORKERS}" =~ ^[1-9][0-9]*$ && "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "HF_DOWNLOAD_WORKERS and HF_DOWNLOAD_ATTEMPTS must be positive integers" >&2
    exit 2
fi

mkdir -p "${DATASET_ROOT}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    echo "Downloading pinned G1 multitask dataset (attempt ${attempt}/${MAX_ATTEMPTS})..."
    if hf download "${GR00T_G1_MULTITASK_DATASET}" \
        --type dataset \
        --revision "${GR00T_G1_MULTITASK_DATASET_REVISION}" \
        --local-dir "${DATASET_ROOT}" \
        --max-workers "${DOWNLOAD_WORKERS}"; then
        hf cache verify "${GR00T_G1_MULTITASK_DATASET}" \
            --type dataset \
            --revision "${GR00T_G1_MULTITASK_DATASET_REVISION}" \
            --local-dir "${DATASET_ROOT}" \
            --fail-on-missing-files
        "${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
            "${PROJECT_ROOT}/scripts/multitask/audit_dataset.py" \
            --root "${DATASET_ROOT}" --strict \
            --json-output "${PROJECT_ROOT}/outputs/g1_multitask_dataset_audit.json"
        exit 0
    fi
    if (( attempt == MAX_ATTEMPTS )); then
        break
    fi
    delay=$((attempt * 15))
    echo "Hub throttled or interrupted the transfer; resuming in ${delay}s. Logging in with 'hf auth login' raises the rate limit."
    sleep "${delay}"
done

echo "Dataset download is still incomplete after ${MAX_ATTEMPTS} attempts; rerun the same command to resume." >&2
exit 1
