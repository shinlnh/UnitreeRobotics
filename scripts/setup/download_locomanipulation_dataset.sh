#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

DATA_ROOT="${GR00T_LOCOMANIP_DATASET_PATH:-${PROJECT_ROOT}/datasets/g1_locomanip_hf}"
ARCHIVE="${DATA_ROOT}/${GR00T_LOCOMANIP_DATASET_ARCHIVE}"
INITIAL_STATE_DATASET="${DATA_ROOT}/dataset_annotated_g1_locomanip.hdf5"
HF=(uv tool run --from "huggingface_hub==${HF_HUB_CLI_VERSION}" hf)

mkdir -p "${DATA_ROOT}"
"${HF[@]}" download "${GR00T_LOCOMANIP_DATASET}" \
    "${GR00T_LOCOMANIP_DATASET_ARCHIVE}" README.md .gitattributes \
    --repo-type dataset \
    --revision "${GR00T_LOCOMANIP_DATASET_REVISION}" \
    --local-dir "${DATA_ROOT}" \
    --max-workers "${HF_DOWNLOAD_WORKERS:-2}"
"${HF[@]}" cache verify "${GR00T_LOCOMANIP_DATASET}" \
    --repo-type dataset \
    --revision "${GR00T_LOCOMANIP_DATASET_REVISION}" \
    --local-dir "${DATA_ROOT}" \
    --fail-on-missing-files

if [[ ! -f "${DATA_ROOT}/g1_simple_high_var_lerobot/meta/modality.json" ]]; then
    unzip -q "${ARCHIVE}" -d "${DATA_ROOT}"
fi
if [[ ! -s "${INITIAL_STATE_DATASET}" ]]; then
    curl -L --fail --retry 3 --output "${INITIAL_STATE_DATASET}" \
        "${GR00T_LOCOMANIP_INITIAL_STATE_URL}"
fi
test -f "${DATA_ROOT}/g1_simple_high_var_lerobot/meta/modality.json"
test -s "${INITIAL_STATE_DATASET}"
echo "Verified locomanipulation training dataset: ${DATA_ROOT}/g1_simple_high_var_lerobot"
echo "Verified rollout initial-state dataset: ${INITIAL_STATE_DATASET}"
