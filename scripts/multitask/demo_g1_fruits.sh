#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${MULTITASK_TMUX_SESSION:-unitree_g1_multitask}"
PORT="${POLICY_PORT:-5550}"
HEADLESS="${MULTITASK_HEADLESS:-0}"
STEPS="${MULTITASK_STEPS:-3000}"
INSTRUCTION="${INSTRUCTION:-Pick up the red apple and place it on the plate}"
MODE="${GR00T_MULTITASK_MODE:-base}"

[[ "${HEADLESS}" == "0" || "${HEADLESS}" == "1" ]] || {
    echo "MULTITASK_HEADLESS must be 0 or 1" >&2
    exit 2
}
tmux kill-session -t "${SESSION}" 2>/dev/null || true
printf -v root_q '%q' "${PROJECT_ROOT}"
tmux new-session -d -s "${SESSION}" -n policy \
    "cd ${root_q} && POLICY_PORT=${PORT} GR00T_MULTITASK_MODE=${MODE} exec bash scripts/multitask/serve_g1_fruits.sh"

ready=0
for _ in $(seq 1 180); do
    if "${PROJECT_ROOT}/.deps/Isaac-GR00T/.venv/bin/python" - <<PY >/dev/null 2>&1
import socket
s = socket.socket(); s.settimeout(0.2)
raise SystemExit(0 if s.connect_ex(("127.0.0.1", ${PORT})) == 0 else 1)
PY
    then
        ready=1
        break
    fi
    if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
        break
    fi
    sleep 1
done
if [[ "${ready}" != "1" ]]; then
    echo "GR00T server did not become ready. Policy log:" >&2
    tmux capture-pane -pt "${SESSION}:policy" 2>/dev/null || true
    exit 1
fi

demo_args=(--port "${PORT}" --steps "${STEPS}" --instruction "${INSTRUCTION}")
if [[ "${MODE}" == "base" ]]; then
    demo_args+=(--policy-contract base-real-g1)
elif [[ "${MODE}" == "finetuned" ]]; then
    demo_args+=(--policy-contract posttrain-43d)
else
    echo "GR00T_MULTITASK_MODE must be 'base' or 'finetuned'" >&2
    exit 2
fi
[[ "${HEADLESS}" == "1" ]] && demo_args+=(--headless)
printf -v demo_args_q ' %q' "${demo_args[@]}"
tmux new-window -d -t "${SESSION}" -n simulation \
    "cd ${root_q}/.deps/GR00T-WholeBodyControl && MUJOCO_GL=$([[ ${HEADLESS} == 1 ]] && printf egl || printf glx) exec .venv_sim/bin/python ${root_q}/scripts/multitask/run_g1_fruits_demo.py${demo_args_q}"
tmux select-window -t "${SESSION}:simulation"
exec tmux attach-session -t "${SESSION}"
