#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA_DIR="$PROJECT_ROOT/.upstream/IsaacLab-Arena"
GROOT_DIR="$ARENA_DIR/submodules/Isaac-GR00T"
MODEL_DIR="$PROJECT_ROOT/checkpoints/arena/loco_box"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/shin/IsaacLab}"
RUN_ROOT="$PROJECT_ROOT/artifacts/isaac_arena/loco_box"
STATE_DIR="$RUN_ROOT/state"
SERVER_LOG="$RUN_ROOT/gr00t_server.log"
CLIENT_LOG="$RUN_ROOT/isaac_headless.log"
SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"
SERVER_PID_FILE="$STATE_DIR/server.pid"
CLIENT_PID_FILE="$STATE_DIR/client.pid"
SUPERVISOR_PID_FILE="$STATE_DIR/supervisor.pid"

mkdir -p "$STATE_DIR"

pid_is_live() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null
}

require_layout() {
    local required
    for required in \
        "$ARENA_DIR/isaaclab_arena/evaluation/policy_runner.py" \
        "$GROOT_DIR/gr00t/eval/run_gr00t_server.py" \
        "$GROOT_DIR/.venv/bin/python" \
        "$MODEL_DIR/config.json" \
        "$MODEL_DIR/model-00001-of-00002.safetensors" \
        "$MODEL_DIR/model-00002-of-00002.safetensors" \
        "$ISAACLAB_DIR/.venv/bin/python" \
        "$ISAACLAB_DIR/_isaac_sim/setup_python_env.sh"; do
        if [[ ! -e "$required" ]]; then
            echo "Missing required path: $required" >&2
            exit 2
        fi
    done
}

server_fg() {
    require_layout
    cd "$GROOT_DIR"
    export NO_ALBUMENTATIONS_UPDATE=1
    export PYTHONUNBUFFERED=1
    exec "$GROOT_DIR/.venv/bin/python" gr00t/eval/run_gr00t_server.py \
        --modality-config-path "$ARENA_DIR/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py" \
        --model-path "$MODEL_DIR" \
        --embodiment-tag NEW_EMBODIMENT \
        --device cuda \
        --host 127.0.0.1 \
        --port 5555
}

client_fg() {
    require_layout
    cd "$ARENA_DIR"

    local arena_lab_paths
    arena_lab_paths="$(find "$ARENA_DIR/submodules/IsaacLab/source" \
        -mindepth 1 -maxdepth 1 -type d -name 'isaaclab*' -printf '%p:')"

    export VIRTUAL_ENV="$ISAACLAB_DIR/.venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export ISAAC_PATH="$ISAACLAB_DIR/_isaac_sim"
    export CARB_APP_PATH="$ISAAC_PATH/kit"
    export EXP_PATH="$ISAAC_PATH/apps"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="$GROOT_DIR:${arena_lab_paths}${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    # shellcheck disable=SC1091
    source "$ISAAC_PATH/setup_python_env.sh"

    exec "$VIRTUAL_ENV/bin/python" isaaclab_arena/evaluation/policy_runner.py \
        --viz none \
        --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml \
        --remote_host 127.0.0.1 \
        --remote_port 5555 \
        --num_steps 1500 \
        --enable_cameras \
        --record_camera_video \
        --output_base_dir "$RUN_ROOT/outputs" \
        galileo_g1_locomanip_pick_and_place \
        --object brown_box \
        --embodiment g1_wbc_joint
}

supervisor_fg() {
    require_layout

    if ! pid_is_live "$SERVER_PID_FILE"; then
        : >"$SERVER_LOG"
        "$0" server-fg >>"$SERVER_LOG" 2>&1 &
        echo "$!" >"$SERVER_PID_FILE"
    fi

    if ! PYTHONPATH="$GROOT_DIR:$ARENA_DIR:${PYTHONPATH:-}" \
        "$GROOT_DIR/.venv/bin/python" \
        "$ARENA_DIR/isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py" \
        --host 127.0.0.1 --port 5555 \
        --timeout-sec 360 --poll-interval-sec 2 --request-timeout-ms 2000; then
        echo "GR00T server did not answer ping. See $SERVER_LOG" >&2
        exit 4
    fi

    : >"$CLIENT_LOG"
    "$0" client-fg >>"$CLIENT_LOG" 2>&1 &
    echo "$!" >"$CLIENT_PID_FILE"
    wait "$(<"$CLIENT_PID_FILE")"
}

start() {
    require_layout
    if pid_is_live "$SUPERVISOR_PID_FILE"; then
        echo "Loco-box demo is already supervised by PID $(<"$SUPERVISOR_PID_FILE")."
        exit 0
    fi

    : >"$SUPERVISOR_LOG"
    nohup setsid "$0" supervisor-fg >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
    echo "$!" >"$SUPERVISOR_PID_FILE"
    echo "Started detached loco-box demo (supervisor PID $!)."
    echo "Status: $0 status"
}

status() {
    local label pid_file state pid label_pid
    for label_pid in \
        "supervisor:$SUPERVISOR_PID_FILE" \
        "server:$SERVER_PID_FILE" \
        "client:$CLIENT_PID_FILE"; do
        label="${label_pid%%:*}"
        pid_file="${label_pid#*:}"
        if pid_is_live "$pid_file"; then
            pid="$(<"$pid_file")"
            state="running"
        elif [[ -s "$pid_file" ]]; then
            pid="$(<"$pid_file")"
            state="exited"
        else
            pid="-"
            state="not-started"
        fi
        printf '%-10s %-12s pid=%s\n' "$label" "$state" "$pid"
    done

    echo "Logs: $RUN_ROOT"
    find "$RUN_ROOT/outputs" -type f \( -name '*.mp4' -o -name '*.jsonl' -o -name 'index.html' \) \
        -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true
}

case "${1:-start}" in
    start) start ;;
    status) status ;;
    server-fg) server_fg ;;
    client-fg) client_fg ;;
    supervisor-fg) supervisor_fg ;;
    *)
        echo "Usage: $0 {start|status}" >&2
        exit 2
        ;;
esac
