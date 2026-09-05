#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA_DIR="$PROJECT_ROOT/.upstream/IsaacLab-Arena"
GROOT_DIR="$PROJECT_ROOT/.upstream/Isaac-GR00T-N1.7"
MODEL_DIR="$PROJECT_ROOT/checkpoints/arena/static_apple"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/shin/IsaacLab}"
ARENA_RUN_NAME="${ARENA_RUN_NAME:-static_apple}"
ARENA_OBJECT="${ARENA_OBJECT:-apple_01_objaverse_robolab}"
ARENA_LANGUAGE_INSTRUCTION="${ARENA_LANGUAGE_INSTRUCTION:-}"
ARENA_VIZ="${ARENA_VIZ:-none}"
ARENA_NUM_STEPS="${ARENA_NUM_STEPS:-600}"
ARENA_DEMO_ONCE="${ARENA_DEMO_ONCE:-0}"
ARENA_POST_COMPLETION_STEPS="${ARENA_POST_COMPLETION_STEPS:-0}"
ARENA_CONTINUOUS_RUNTIME="${ARENA_CONTINUOUS_RUNTIME:-0}"
ARENA_RECORD_VIDEO="${ARENA_RECORD_VIDEO:-1}"
ARENA_REMOTE_TIMEOUT_MS="${ARENA_REMOTE_TIMEOUT_MS:-120000}"
ARENA_SEED="${ARENA_SEED:-42}"
ARENA_MODEL_SEED="${ARENA_MODEL_SEED:-42}"
ARENA_ASSET_CACHE_DIR="${ARENA_ASSET_CACHE_DIR:-$PROJECT_ROOT/.cache/isaaclab-assets}"
RUN_ROOT="$PROJECT_ROOT/artifacts/isaac_arena/$ARENA_RUN_NAME"
STATE_DIR="$RUN_ROOT/state"
SERVER_LOG="$RUN_ROOT/gr00t_server.log"
CLIENT_LOG="$RUN_ROOT/isaac_headless.log"
SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"
SERVER_PID_FILE="$STATE_DIR/server.pid"
CLIENT_PID_FILE="$STATE_DIR/client.pid"
SUPERVISOR_PID_FILE="$STATE_DIR/supervisor.pid"
GUI_SERVER_PID=""
GUI_CLIENT_PID=""

mkdir -p "$STATE_DIR" "$ARENA_ASSET_CACHE_DIR"

pid_is_live() {
    local pid_file="$1"
    [[ -s "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null
}

terminate_process_group() {
    local process_pid="$1"
    local attempt
    [[ "$process_pid" =~ ^[0-9]+$ ]] || return 0
    kill -0 "$process_pid" 2>/dev/null || return 0

    kill -TERM -- "-$process_pid" 2>/dev/null || kill -TERM "$process_pid" 2>/dev/null || true
    for attempt in {1..20}; do
        kill -0 "$process_pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$process_pid" 2>/dev/null; then
        kill -KILL -- "-$process_pid" 2>/dev/null || kill -KILL "$process_pid" 2>/dev/null || true
    fi
    wait "$process_pid" 2>/dev/null || true
}

cleanup_gui() {
    terminate_process_group "$GUI_CLIENT_PID"
    terminate_process_group "$GUI_SERVER_PID"
    GUI_CLIENT_PID=""
    GUI_SERVER_PID=""
    rm -f "$CLIENT_PID_FILE" "$SERVER_PID_FILE"
}

require_layout() {
    local required
    for required in \
        "$ARENA_DIR/isaaclab_arena/evaluation/policy_runner.py" \
        "$GROOT_DIR/gr00t/eval/run_gr00t_server.py" \
        "$MODEL_DIR/config.json" \
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
    exec uv run python gr00t/eval/run_gr00t_server.py \
        --modality-config-path "$ARENA_DIR/isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_gr00t_n_1_7_config.py" \
        --model-path "$MODEL_DIR" \
        --embodiment-tag NEW_EMBODIMENT \
        --device cuda \
        --seed "$ARENA_MODEL_SEED" \
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
    # Keep recursively downloaded USD/texture dependencies across reboots.
    export ISAACLAB_ASSET_CACHE_DIR="$ARENA_ASSET_CACHE_DIR"
    # Isaac Sim's environment script requires bash because it uses BASH_SOURCE.
    # shellcheck disable=SC1091
    source "$ISAAC_PATH/setup_python_env.sh"

    local language_args=()
    if [[ -n "$ARENA_LANGUAGE_INSTRUCTION" ]]; then
        language_args=(--language_instruction "$ARENA_LANGUAGE_INSTRUCTION")
    fi
    local recording_args=()
    if [[ "$ARENA_RECORD_VIDEO" == "1" ]]; then
        recording_args=(--record_camera_video)
    fi
    local length_args=(--num_steps "$ARENA_NUM_STEPS")
    local completion_args=()
    if [[ "$ARENA_DEMO_ONCE" == "1" ]]; then
        length_args=(--num_episodes 1)
        completion_args=(--hold_on_completion)
        if [[ "$ARENA_CONTINUOUS_RUNTIME" == "1" ]]; then
            completion_args+=(--continuous_runtime)
        elif (( ARENA_POST_COMPLETION_STEPS > 0 )); then
            completion_args+=(--post_completion_steps "$ARENA_POST_COMPLETION_STEPS")
        fi
    fi

    exec "$VIRTUAL_ENV/bin/python" isaaclab_arena/evaluation/policy_runner.py \
        --viz "$ARENA_VIZ" \
        --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy \
        --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/g1_static_apple_gr00t_closedloop_config.yaml \
        --remote_host 127.0.0.1 \
        --remote_port 5555 \
        --remote_timeout_ms "$ARENA_REMOTE_TIMEOUT_MS" \
        --seed "$ARENA_SEED" \
        "${length_args[@]}" \
        "${completion_args[@]}" \
        --enable_cameras \
        "${recording_args[@]}" \
        --output_base_dir "$RUN_ROOT/outputs" \
        "${language_args[@]}" \
        galileo_g1_static_pick_and_place \
        --object "$ARENA_OBJECT" \
        --destination clay_plates_hot3d_robolab \
        --embodiment g1_wbc_agile_joint
}

gui() {
    require_layout
    export ARENA_VIZ=kit
    # The live viewport is already visible; skipping MP4 capture saves GPU memory
    # and avoids growing artifacts during long interactive sessions.
    export ARENA_RECORD_VIDEO=0

    local existing_client existing_server
    existing_client="$(pgrep -f '[i]saaclab_arena/evaluation/policy_runner.py' | head -n 1 || true)"
    existing_server="$(pgrep -f '[r]un_gr00t_server.py' | head -n 1 || true)"
    if [[ -n "$existing_client" || -n "$existing_server" ]]; then
        echo "Another Isaac Arena/GR00T demo is already running; refusing to open a second instance." >&2
        echo "client_pid=${existing_client:--} server_pid=${existing_server:--}" >&2
        return 3
    fi

    if pid_is_live "$SERVER_PID_FILE" || pid_is_live "$CLIENT_PID_FILE"; then
        echo "An Arena process is already running. Check it with: $0 status" >&2
        return 3
    fi

    : >"$SERVER_LOG"
    setsid "$0" server-fg >>"$SERVER_LOG" 2>&1 &
    GUI_SERVER_PID=$!
    echo "$GUI_SERVER_PID" >"$SERVER_PID_FILE"
    trap 'cleanup_gui; exit 130' INT TERM
    trap cleanup_gui EXIT

    if ! PYTHONPATH="$GROOT_DIR:$ARENA_DIR:${PYTHONPATH:-}" \
        "$GROOT_DIR/.venv/bin/python" \
        "$ARENA_DIR/isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py" \
        --host 127.0.0.1 --port 5555 \
        --timeout-sec 360 --poll-interval-sec 2 --request-timeout-ms 2000; then
        echo "GR00T server did not answer ping. See $SERVER_LOG" >&2
        return 4
    fi

    echo "Opening Isaac Sim GUI. Close with Ctrl+C in this terminal."
    setsid "$0" client-fg &
    GUI_CLIENT_PID=$!
    echo "$GUI_CLIENT_PID" >"$CLIENT_PID_FILE"

    local client_exit=0
    wait "$GUI_CLIENT_PID" || client_exit=$?
    cleanup_gui
    trap - EXIT INT TERM
    return "$client_exit"
}

demo_once() {
    export ARENA_DEMO_ONCE=1
    gui
}

observe_continuous() {
    export ARENA_DEMO_ONCE=1
    export ARENA_CONTINUOUS_RUNTIME=1
    gui
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
        echo "Arena run '$ARENA_RUN_NAME' is already supervised by PID $(<"$SUPERVISOR_PID_FILE")."
        exit 0
    fi

    : >"$SUPERVISOR_LOG"
    nohup setsid "$0" supervisor-fg >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
    echo "$!" >"$SUPERVISOR_PID_FILE"
    echo "Started detached Arena run '$ARENA_RUN_NAME' with object '$ARENA_OBJECT' (supervisor PID $!)."
    echo "Status: $0 status"
}

status() {
    local label pid_file state pid
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
    gui) gui ;;
    demo-once) demo_once ;;
    observe-continuous|observe-after-success) observe_continuous ;;
    status) status ;;
    server-fg) server_fg ;;
    client-fg) client_fg ;;
    supervisor-fg) supervisor_fg ;;
    *)
        echo "Usage: $0 {start|gui|demo-once|observe-continuous|observe-after-success|status}" >&2
        exit 2
        ;;
esac
