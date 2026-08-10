#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="${APPLEPNP_TMUX_SESSION:-unitree_applepnp}"
PORT="${APPLEPNP_PORT:-5550}"
HEADLESS="${APPLEPNP_HEADLESS:-0}"
STEPS="${APPLEPNP_STEPS:-3000}"

tmux kill-session -t "${SESSION}" 2>/dev/null || true
printf -v root_q '%q' "${PROJECT_ROOT}"
tmux new-session -d -s "${SESSION}" -n policy \
    "cd ${root_q} && exec bash scripts/groot/run_with_groot_env.sh .deps/Isaac-GR00T/.venv/bin/python scripts/applepnp/serve_applepnp.py --port ${PORT}"

for _ in $(seq 1 120); do
    if "${PROJECT_ROOT}/.deps/Isaac-GR00T/.venv/bin/python" - <<PY >/dev/null 2>&1
import socket
s = socket.socket(); s.settimeout(0.2)
raise SystemExit(0 if s.connect_ex(("127.0.0.1", ${PORT})) == 0 else 1)
PY
    then
        break
    fi
    sleep 1
done

demo_args=(--port "${PORT}" --steps "${STEPS}")
if [[ "${HEADLESS}" == "1" ]]; then
    demo_args+=(--headless)
elif [[ "${HEADLESS}" != "0" ]]; then
    echo "APPLEPNP_HEADLESS must be 0 or 1" >&2
    exit 2
fi
printf -v demo_args_q ' %q' "${demo_args[@]}"
tmux new-window -d -t "${SESSION}" -n simulation \
    "cd ${root_q}/.deps/GR00T-WholeBodyControl && MUJOCO_GL=$([[ ${HEADLESS} == 1 ]] && printf egl || printf glx) exec .venv_sim/bin/python ${root_q}/scripts/applepnp/run_wbc_demo.py${demo_args_q}"
tmux select-window -t "${SESSION}:simulation"
exec tmux attach-session -t "${SESSION}"
