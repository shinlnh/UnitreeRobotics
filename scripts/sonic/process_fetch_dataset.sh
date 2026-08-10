#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
: "${RAW_SONIC_FETCH_DATASET:?Set RAW_SONIC_FETCH_DATASET to the recorded SONIC dataset directory}"
CLEAN_SONIC_FETCH_DATASET="${CLEAN_SONIC_FETCH_DATASET:-${PROJECT_ROOT}/datasets/sonic/g1_fetch_clean}"

[[ -x "${SONIC_DIR}/.venv_data_collection/bin/python" ]] || {
    echo "SONIC data environment is missing. Run: make setup-sonic-data" >&2
    exit 1
}
[[ -f "${RAW_SONIC_FETCH_DATASET}/meta/info.json" ]] || {
    echo "Raw LeRobot dataset not found: ${RAW_SONIC_FETCH_DATASET}" >&2
    exit 1
}
if [[ -e "${CLEAN_SONIC_FETCH_DATASET}" ]]; then
    echo "Refusing to overwrite existing clean dataset: ${CLEAN_SONIC_FETCH_DATASET}" >&2
    exit 1
fi

cd "${SONIC_DIR}"
exec .venv_data_collection/bin/python gear_sonic/scripts/process_dataset.py \
    --dataset-path "${RAW_SONIC_FETCH_DATASET}" \
    --output-path "${CLEAN_SONIC_FETCH_DATASET}"
