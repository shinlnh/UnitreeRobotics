#!/usr/bin/env bash
set -euo pipefail

# Automated unattended transport smoke. It intentionally records a neutral
# episode only to validate camera + controller + exporter wiring. The artifact
# is never copied to datasets/sonic/g1_fetch_clean and is not training data.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/sonic/sonic_deploy_env.sh"

SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
SESSION_NAME="${FETCH_TMUX_SESSION:-unitree_fetch_smoke}"
SMOKE_TIMEOUT="${FETCH_SMOKE_TIMEOUT:-300}"
RECORD_SECONDS="${FETCH_SMOKE_RECORD_SECONDS:-4}"
DATASET_NAME="${FETCH_DATASET_NAME:-g1_fetch_smoke_$(date +%Y%m%d_%H%M%S)}"
DATASET_PATH="${SONIC_DIR}/outputs/${DATASET_NAME}"
LOG_DIR="${FETCH_LOG_DIR:-${PROJECT_ROOT}/logs/sonic/${SESSION_NAME}}"

if ! [[ "${SMOKE_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FETCH_SMOKE_TIMEOUT must be a positive integer" >&2
    exit 2
fi
if ! [[ "${RECORD_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FETCH_SMOKE_RECORD_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ -e "${DATASET_PATH}" ]]; then
    echo "Refusing to reuse an existing smoke artifact: ${DATASET_PATH}" >&2
    exit 1
fi

cleanup() {
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

pane_contains() {
    local window="$1"
    local pattern="$2"
    # Do not use grep -q here: under pipefail it can close the pipe before
    # tmux finishes writing a large pane, turning a real match into SIGPIPE.
    tmux capture-pane -p -t "${SESSION_NAME}:${window}" -S -300 2>/dev/null \
        | grep -F "${pattern}" >/dev/null
}

FETCH_DETACH=1 \
FETCH_SMOKE_MANAGER=1 \
FETCH_TMUX_SESSION="${SESSION_NAME}" \
FETCH_DATASET_NAME="${DATASET_NAME}" \
FETCH_SCENE_SEED="${FETCH_SCENE_SEED:-123}" \
    bash "${PROJECT_ROOT}/scripts/sonic/collect_fetch.sh"

echo "Waiting for SONIC CONTROL and exporter readiness (timeout: ${SMOKE_TIMEOUT}s)..."
ready=0
for ((elapsed = 0; elapsed < SMOKE_TIMEOUT; elapsed++)); do
    if pane_contains deploy "transitioning to CONTROL" \
        && pane_contains exporter "Recording to"; then
        ready=1
        break
    fi
    if ! tmux list-windows -t "${SESSION_NAME}" -F '#{window_name}' 2>/dev/null \
        | grep -Fx exporter >/dev/null; then
        echo "The exporter exited before the smoke stack became ready." >&2
        [[ ! -f "${LOG_DIR}/exporter.log" ]] || tail -100 "${LOG_DIR}/exporter.log" >&2
        exit 1
    fi
    sleep 1
done
if [[ "${ready}" != "1" ]]; then
    echo "SONIC did not enter CONTROL within ${SMOKE_TIMEOUT}s." >&2
    exit 1
fi

tmux send-keys -t "${SESSION_NAME}:keyboard" c Enter
sleep "${RECORD_SECONDS}"
tmux send-keys -t "${SESSION_NAME}:keyboard" c Enter

saved=0
for ((elapsed = 0; elapsed < 60; elapsed++)); do
    if pane_contains exporter "Finished saving episode"; then
        saved=1
        break
    fi
    sleep 1
done
if [[ "${saved}" != "1" ]]; then
    echo "Exporter did not save the smoke episode within 60s." >&2
    exit 1
fi

"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/sonic/audit_fetch_dataset.py" \
    --dataset "${DATASET_PATH}" --min-episodes 1 --min-prompts 1 --strict

echo "Fetch transport smoke passed: ${DATASET_PATH}"
echo "The neutral smoke artifact is diagnostic only; do not process or train on it."
