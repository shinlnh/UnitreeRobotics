#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/sonic/sonic_deploy_env.sh"
SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"

if [[ -n "${SONIC_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "${SONIC_LOG_FILE}")"
    exec > >(tee -a "${SONIC_LOG_FILE}") 2>&1
fi
export PYTHONUNBUFFERED=1

cd "${SONIC_DIR}"
exec .venv_data_collection/bin/python gear_sonic/scripts/run_data_exporter.py "$@"
