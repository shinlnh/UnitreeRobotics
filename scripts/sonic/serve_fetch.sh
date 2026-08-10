#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"
GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
: "${GR00T_FETCH_CHECKPOINT:?Set GR00T_FETCH_CHECKPOINT to a finetuned checkpoint directory}"
POLICY_PORT="${POLICY_PORT:-5550}"

[[ -x "${GR00T_DIR}/.venv/bin/python" ]] || {
    echo "GR00T environment is incomplete. Run: make setup-groot" >&2
    exit 1
}
[[ -d "${GR00T_FETCH_CHECKPOINT}" ]] || {
    echo "Checkpoint directory does not exist: ${GR00T_FETCH_CHECKPOINT}" >&2
    exit 1
}

cd "${GR00T_DIR}"
exec .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path "${GR00T_FETCH_CHECKPOINT}" \
    --embodiment-tag UNITREE_G1_SONIC \
    --device cuda:0 \
    --port "${POLICY_PORT}"
