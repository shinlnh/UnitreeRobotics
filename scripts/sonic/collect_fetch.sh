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
FETCH_DATASET_NAME="${FETCH_DATASET_NAME:-g1_fetch}"
FETCH_MODE="${FETCH_MODE:-sim}"
PICO_INPUT_SOURCE="${PICO_INPUT_SOURCE:-xrt}"
SESSION_NAME="${FETCH_TMUX_SESSION:-unitree_fetch_collect}"
RECORD_WRIST_CAMERAS="${RECORD_WRIST_CAMERAS:-$([[ "${FETCH_MODE}" == "real" ]] && printf 1 || printf 0)}"
FETCH_DETACH="${FETCH_DETACH:-0}"
FETCH_SMOKE_MANAGER="${FETCH_SMOKE_MANAGER:-0}"
FETCH_LOG_DIR="${FETCH_LOG_DIR:-${PROJECT_ROOT}/logs/sonic/${SESSION_NAME}}"

args=(
    --task-prompt "${FETCH_PROMPT}"
    --dataset-name "${FETCH_DATASET_NAME}"
    --data-exporter-frequency 50
    --deploy-checkpoint policy/sonic_v1_1/model
    --deploy-obs-config policy/sonic_v1_1/observation_config.yaml
    --pico-input-source "${PICO_INPUT_SOURCE}"
)
if [[ "${RECORD_WRIST_CAMERAS}" == "1" ]]; then
    args+=(--record-wrist-cameras)
elif [[ "${RECORD_WRIST_CAMERAS}" != "0" ]]; then
    echo "RECORD_WRIST_CAMERAS must be 0 or 1" >&2
    exit 1
fi
if [[ "${FETCH_MODE}" == "sim" ]]; then
    :
elif [[ "${FETCH_MODE}" != "real" ]]; then
    echo "FETCH_MODE must be sim or real" >&2
    exit 1
fi

"${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/sonic/preflight.py" \
    --require "$([[ "${FETCH_MODE}" == "sim" ]] && printf collection-sim || printf data)" \
    --strict

cd "${SONIC_DIR}"
if [[ "${FETCH_MODE}" == "real" ]]; then
    exec .venv_data_collection/bin/python gear_sonic/scripts/launch_data_collection.py "${args[@]}" "$@"
fi
if (($#)); then
    echo "Simulation collection does not accept passthrough args; use environment variables." >&2
    exit 2
fi

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
printf -v project_q '%q' "${PROJECT_ROOT}"
printf -v sonic_q '%q' "${SONIC_DIR}"
printf -v prompt_q '%q' "${FETCH_PROMPT}"
printf -v dataset_q '%q' "${FETCH_DATASET_NAME}"
printf -v pico_source_q '%q' "${PICO_INPUT_SOURCE}"
printf -v object_q '%q' "${FETCH_OBJECT}"
printf -v source_q '%q' "${FETCH_SOURCE}"
printf -v destination_q '%q' "${FETCH_DESTINATION}"
printf -v object_profile_q '%q' "${FETCH_OBJECT_PROFILE}"
printf -v source_profile_q '%q' "${FETCH_SOURCE_PROFILE}"
printf -v destination_profile_q '%q' "${FETCH_DESTINATION_PROFILE}"
printf -v scene_seed_q '%q' "${FETCH_SCENE_SEED}"
mkdir -p "${FETCH_LOG_DIR}"
printf -v exporter_log_q '%q' "${FETCH_LOG_DIR}/exporter.log"

tmux new-session -d -s "${SESSION_NAME}" -n sim \
    "cd ${sonic_q} && exec .venv_sim/bin/python ${project_q}/scripts/sonic/run_fetch_sim_loop.py --object-profile ${object_profile_q} --source-profile ${source_profile_q} --destination-profile ${destination_profile_q} --scene-seed ${scene_seed_q} --enable-image-publish --enable-offscreen --camera-port 5555"
tmux new-window -d -t "${SESSION_NAME}" -n deploy \
    "exec bash ${project_q}/scripts/sonic/run_sonic_deploy.sh"
if [[ "${FETCH_SMOKE_MANAGER}" == "1" ]]; then
    tmux new-window -d -t "${SESSION_NAME}" -n teleop \
        "cd ${sonic_q} && exec .venv_teleop/bin/python ${project_q}/scripts/sonic/smoke_sonic_manager.py"
elif [[ "${FETCH_SMOKE_MANAGER}" == "0" ]]; then
    tmux new-window -d -t "${SESSION_NAME}" -n teleop \
        "cd ${sonic_q} && exec .venv_teleop/bin/python gear_sonic/scripts/pico_manager_thread_server.py --input-source ${pico_source_q} --manager"
else
    echo "FETCH_SMOKE_MANAGER must be 0 or 1" >&2
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    exit 2
fi

exporter_command="SONIC_LOG_FILE=${exporter_log_q} exec bash ${project_q}/scripts/sonic/run_data_exporter.sh --task-prompt ${prompt_q} --dataset-name ${dataset_q} --data-collection-frequency 50 --camera-host localhost --camera-port 5555 --no-text-to-speech"
if [[ "${RECORD_WRIST_CAMERAS}" == "1" ]]; then
    exporter_command+=" --record-wrist-cameras"
fi
tmux new-window -d -t "${SESSION_NAME}" -n exporter "${exporter_command}"
tmux new-window -d -t "${SESSION_NAME}" -n keyboard \
    "cd ${sonic_q} && exec .venv_data_collection/bin/python ${project_q}/scripts/sonic/keyboard_control.py --object-name ${object_q} --source ${source_q} --destination ${destination_q}"

tmux select-window -t "${SESSION_NAME}:deploy"
echo "Fetch collection scene ready in tmux session ${SESSION_NAME}."
echo "Switch windows with Ctrl-b n. Keyboard: c=record/save, x=discard, r=reset/randomize."
if [[ "${FETCH_DETACH}" == "1" ]]; then
    exit 0
elif [[ "${FETCH_DETACH}" != "0" ]]; then
    echo "FETCH_DETACH must be 0 or 1" >&2
    exit 2
fi
exec tmux attach-session -t "${SESSION_NAME}"
