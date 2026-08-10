#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"
GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
MODEL_ROOT="${PROJECT_ROOT}/checkpoints/groot-g1-navigation"
MODEL_PATH="${GR00T_CHECKPOINT:-}"
SERVER_HOST="${GR00T_SERVER_BIND:-0.0.0.0}"
SERVER_PORT="${GR00T_SERVER_PORT:-5555}"

[[ -x "${GR00T_DIR}/.venv/bin/python" ]] || {
    echo "Run scripts/setup/bootstrap_groot.sh first." >&2
    exit 1
}
if [[ -z "${MODEL_PATH}" && -d "${MODEL_ROOT}" ]]; then
    MODEL_PATH="$(find "${MODEL_ROOT}" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V | tail -n 1)"
fi
[[ -d "${MODEL_PATH}" ]] || {
    echo "No GR00T checkpoint found. Set GR00T_CHECKPOINT or train below ${MODEL_ROOT}." >&2
    exit 1
}

cd "${GR00T_DIR}"
exec uv run python gr00t/eval/run_gr00t_server.py \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    "$@"
