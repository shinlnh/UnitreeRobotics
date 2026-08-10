#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${MULTITASK_TMUX_SESSION:-unitree_g1_multitask_isaac}"
PORT="${POLICY_PORT:-5550}"
INTERACTIVE="${MULTITASK_INTERACTIVE:-0}"
if [[ "${INTERACTIVE}" == "1" ]]; then
    INSTRUCTION="${INSTRUCTION-}"
else
    INSTRUCTION="${INSTRUCTION:-Pick up the red apple and place it on the plate}"
fi
MAX_SECONDS="${MULTITASK_MAX_SECONDS:-50}"
VIZ="${MULTITASK_VIZ:-kit}"
READY_TIMEOUT="${MULTITASK_READY_TIMEOUT:-900}"
RUNNER_LOG="${MULTITASK_RUNNER_LOG:-${PROJECT_ROOT}/outputs/groot_multitask_runner.log}"

[[ "${GR00T_MULTITASK_MODE:-base}" == "base" ]] || {
    echo "Isaac mobile-manipulation demo requires the N1.7 REAL_G1 base contract (EEF + navigation)." >&2
    echo "The optional 43D fruit post-train contract is evaluated through make evaluate-multitask." >&2
    exit 2
}

runner_args=(
    --policy-port "${PORT}"
    --max-seconds "${MAX_SECONDS}"
    --viz "${VIZ}"
)
if [[ "${INTERACTIVE}" == "1" ]]; then
    runner_args+=(--interactive)
fi
if [[ -n "${INSTRUCTION}" ]]; then
    runner_args+=(--instruction "${INSTRUCTION}")
fi

port_is_ready() {
    "${PROJECT_ROOT}/.deps/Isaac-GR00T/.venv/bin/python" - <<PY >/dev/null 2>&1
import socket
s = socket.socket(); s.settimeout(0.2)
raise SystemExit(0 if s.connect_ex(("127.0.0.1", ${PORT})) == 0 else 1)
PY
}

wait_for_policy() {
    local backend="${1}"
    local elapsed
    for ((elapsed = 0; elapsed < READY_TIMEOUT; elapsed++)); do
        if port_is_ready; then
            return 0
        fi
        if [[ "${backend}" == "tmux" ]]; then
            tmux has-session -t "${SESSION}" 2>/dev/null || return 1
        elif ! kill -0 "${policy_pid}" 2>/dev/null; then
            return 1
        fi
        if ((elapsed > 0 && elapsed % 15 == 0)); then
            echo "[DEMO] Waiting for GR00T N1.7 server (${elapsed}s/${READY_TIMEOUT}s)..."
        fi
        sleep 1
    done
    return 1
}

if command -v tmux >/dev/null 2>&1; then
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
    printf -v root_q '%q' "${PROJECT_ROOT}"
    tmux new-session -d -s "${SESSION}" -n policy \
        "cd ${root_q} && POLICY_PORT=${PORT} GR00T_MULTITASK_MODE=base exec bash scripts/multitask/serve_g1_fruits.sh"

    if ! wait_for_policy tmux; then
        echo "GR00T N1.7 server did not become ready. Policy log:" >&2
        tmux capture-pane -pt "${SESSION}:policy" 2>/dev/null || true
        exit 1
    fi

    printf -v runner_args_q ' %q' "${runner_args[@]}"
    printf -v runner_log_q '%q' "${RUNNER_LOG}"
    tmux new-window -d -t "${SESSION}" -n isaac \
        "cd ${root_q} && export OMNI_KIT_ACCEPT_EULA=1 && set -o pipefail && .deps/IsaacLab/.venv/bin/python scripts/multitask/run_isaac_g1_fruits.py${runner_args_q} 2>&1 | tee ${runner_log_q}"
    tmux select-window -t "${SESSION}:isaac"
    exec tmux attach-session -t "${SESSION}"
fi

mkdir -p "${PROJECT_ROOT}/outputs"
policy_log="${MULTITASK_POLICY_LOG:-${PROJECT_ROOT}/outputs/groot_multitask_policy.log}"
policy_pid=""
cleanup() {
    if [[ -n "${policy_pid}" ]] && kill -0 "${policy_pid}" 2>/dev/null; then
        echo "[DEMO] Stopping GR00T policy server (PID ${policy_pid})..."
        kill "${policy_pid}" 2>/dev/null || true
        wait "${policy_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "[DEMO] tmux is not installed; using the built-in foreground launcher."
echo "[DEMO] Policy log: ${policy_log}"
POLICY_PORT="${PORT}" GR00T_MULTITASK_MODE=base \
    bash "${PROJECT_ROOT}/scripts/multitask/serve_g1_fruits.sh" \
    >"${policy_log}" 2>&1 &
policy_pid=$!

if ! wait_for_policy process; then
    echo "GR00T N1.7 server did not become ready. Last policy log lines:" >&2
    tail -n 100 "${policy_log}" >&2 || true
    exit 1
fi

echo "[DEMO] GR00T server is ready; starting Isaac Sim."
echo "[DEMO] Runner log: ${RUNNER_LOG}"
export OMNI_KIT_ACCEPT_EULA=1
set +e
"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/multitask/run_isaac_g1_fruits.py" \
    "${runner_args[@]}" 2>&1 | tee "${RUNNER_LOG}"
runner_status=${PIPESTATUS[0]}
set -e
exit "${runner_status}"
