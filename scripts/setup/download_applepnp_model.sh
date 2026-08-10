#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

MODEL_PATH="${GR00T_APPLEPNP_PATH:-${PROJECT_ROOT}/checkpoints/nvidia-gr00t-n1.7-applepnp-v1}"
HF=(uv tool run --from "huggingface_hub==${HF_HUB_CLI_VERSION}" hf)

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
"${HF[@]}" download "${GR00T_APPLEPNP_MODEL}" \
    --revision "${GR00T_APPLEPNP_MODEL_REVISION}" \
    --local-dir "${MODEL_PATH}" \
    --max-workers "${HF_DOWNLOAD_WORKERS:-2}"
"${HF[@]}" cache verify "${GR00T_APPLEPNP_MODEL}" \
    --revision "${GR00T_APPLEPNP_MODEL_REVISION}" \
    --local-dir "${MODEL_PATH}" \
    --fail-on-missing-files

test -f "${MODEL_PATH}/exported_leapp.yaml"
echo "Verified ApplePnP deployment bundle: ${MODEL_PATH}"
