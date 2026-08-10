#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/sonic/sonic_deploy_env.sh"
SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
FETCH_OBJECT="${FETCH_OBJECT:-apple}"
FETCH_SOURCE="${FETCH_SOURCE:-source table}"
FETCH_DESTINATION="${FETCH_DESTINATION:-green plate}"
FETCH_OBJECT_PROFILE="${FETCH_OBJECT_PROFILE:-apple}"
FETCH_SOURCE_PROFILE="${FETCH_SOURCE_PROFILE:-table}"
FETCH_DESTINATION_PROFILE="${FETCH_DESTINATION_PROFILE:-green_plate}"
FETCH_SCENE_SEED="${FETCH_SCENE_SEED:-0}"
FETCH_PROMPT="${FETCH_PROMPT:-Find the ${FETCH_OBJECT} at the ${FETCH_SOURCE}, pick it up, carry it to the ${FETCH_DESTINATION}, place it safely, and verify delivery.}"
POLICY_HOST="${POLICY_HOST:-localhost}"
POLICY_PORT="${POLICY_PORT:-5550}"
SESSION_NAME="${FETCH_TMUX_SESSION:-unitree_fetch_demo}"
FETCH_DETACH="${FETCH_DETACH:-0}"

"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/sonic/preflight.py" --require demo-sim --strict

cd "${SONIC_DIR}"

if (($#)); then
    echo "demo_fetch_sim.sh does not accept passthrough arguments; use FETCH_PROMPT/POLICY_HOST/POLICY_PORT" >&2
    exit 2
fi

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

printf -v project_q '%q' "${PROJECT_ROOT}"
printf -v sonic_q '%q' "${SONIC_DIR}"
printf -v prompt_q '%q' "${FETCH_PROMPT}"
printf -v host_q '%q' "${POLICY_HOST}"
printf -v port_q '%q' "${POLICY_PORT}"
printf -v object_q '%q' "${FETCH_OBJECT}"
printf -v source_q '%q' "${FETCH_SOURCE}"
printf -v destination_q '%q' "${FETCH_DESTINATION}"
printf -v object_profile_q '%q' "${FETCH_OBJECT_PROFILE}"
printf -v source_profile_q '%q' "${FETCH_SOURCE_PROFILE}"
printf -v destination_profile_q '%q' "${FETCH_DESTINATION_PROFILE}"
printf -v scene_seed_q '%q' "${FETCH_SCENE_SEED}"

tmux new-session -d -s "${SESSION_NAME}" -n sim \
    "cd ${sonic_q} && exec .venv_sim/bin/python ${project_q}/scripts/sonic/run_fetch_sim_loop.py --object-profile ${object_profile_q} --source-profile ${source_profile_q} --destination-profile ${destination_profile_q} --scene-seed ${scene_seed_q} --enable-image-publish --enable-offscreen --camera-port 5555"

tmux new-window -d -t "${SESSION_NAME}" -n deploy \
    "exec bash ${project_q}/scripts/sonic/run_sonic_deploy.sh"

tmux new-window -d -t "${SESSION_NAME}" -n inference \
    "cd ${sonic_q} && exec .venv_inference/bin/python ${project_q}/scripts/sonic/run_vla_inference.py --host ${host_q} --port ${port_q} --embodiment-tag unitree_g1_sonic --prompt ${prompt_q} --action-publish-rate 50 --action-horizon 40 --camera-host localhost --camera-port 5555"

tmux new-window -d -t "${SESSION_NAME}" -n keyboard \
    "cd ${sonic_q} && exec .venv_inference/bin/python ${project_q}/scripts/sonic/keyboard_control.py --object-name ${object_q} --source ${source_q} --destination ${destination_q}"

tmux select-window -t "${SESSION_NAME}:deploy"
echo "Fetch scene ready in tmux session ${SESSION_NAME}."
echo "Switch to keyboard (Ctrl-b n), then send k, i, p."
if [[ "${FETCH_DETACH}" == "1" ]]; then
    exit 0
elif [[ "${FETCH_DETACH}" != "0" ]]; then
    echo "FETCH_DETACH must be 0 or 1" >&2
    exit 2
fi
exec tmux attach-session -t "${SESSION_NAME}"
