#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

MODEL_ROOT="${GR00T_LOCOMANIP_MODEL_PATH:-${PROJECT_ROOT}/checkpoints/g1_locomanip_finetune_hf}"
ARCHIVE="${MODEL_ROOT}/${GR00T_LOCOMANIP_MODEL_ARCHIVE}"
CHECKPOINT="${MODEL_ROOT}/g1_locomanip_finetune_20260129_231610/checkpoint-20000"
HF=(uv tool run --from "huggingface_hub==${HF_HUB_CLI_VERSION}" hf)

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }
mkdir -p "${MODEL_ROOT}"
"${HF[@]}" download "${GR00T_LOCOMANIP_MODEL}" "${GR00T_LOCOMANIP_MODEL_ARCHIVE}" README.md \
    --revision "${GR00T_LOCOMANIP_MODEL_REVISION}" \
    --local-dir "${MODEL_ROOT}" \
    --max-workers "${HF_DOWNLOAD_WORKERS:-2}"
"${HF[@]}" cache verify "${GR00T_LOCOMANIP_MODEL}" \
    --revision "${GR00T_LOCOMANIP_MODEL_REVISION}" \
    --local-dir "${MODEL_ROOT}" \
    --fail-on-missing-files

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
    # Inference needs weights/config/stats only.  Leave the 4.1 GB optimizer
    # and trainer-resume state compressed in the archive.
    unzip -q "${ARCHIVE}" -d "${MODEL_ROOT}" \
        "g1_locomanip_finetune_20260129_231610/checkpoint-20000/config.json" \
        "g1_locomanip_finetune_20260129_231610/checkpoint-20000/model-*.safetensors" \
        "g1_locomanip_finetune_20260129_231610/checkpoint-20000/model.safetensors.index.json" \
        "g1_locomanip_finetune_20260129_231610/checkpoint-20000/experiment_cfg/*"
fi
test -f "${CHECKPOINT}/config.json"
test -f "${CHECKPOINT}/experiment_cfg/metadata.json"
echo "Verified G1 locomanipulation checkpoint: ${CHECKPOINT}"
