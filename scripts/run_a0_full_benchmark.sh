#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ID="${ROBOCEREBRA_EXPERIMENT_ID:-A0}"
VARIANT="${ROBOCEREBRA_VARIANT:-GR00T-N1.7-LIBERO-original}"
CHECKPOINT_PATH="${ROBOCEREBRA_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/robocerebra/GR00T-N1.7-LIBERO/libero_10}"
MODEL_REVISION="${ROBOCEREBRA_MODEL_REVISION:-2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21}"
ARTIFACT_ROOT="${ROBOCEREBRA_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/${EXPERIMENT_ID}/full-benchmark}"
SHARD_ROOT="${ARTIFACT_ROOT}/shards"
BASE_SEED=7
TRIALS=10
POLICY_PORT=5550
SERVER_PID=""

mkdir -p "${ARTIFACT_ROOT}" "${SHARD_ROOT}"

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

start_server() {
  local label="$1"
  local server_log="${ARTIFACT_ROOT}/server-${label}.log"
  if ss -ltn "( sport = :${POLICY_PORT} )" | tail -n +2 | grep -q .; then
    echo "Port ${POLICY_PORT} is already in use; refusing to attach to an unknown server." >&2
    exit 1
  fi
  (
    cd "${PROJECT_ROOT}/.upstream/Isaac-GR00T-N1.7"
    exec setsid uv run --no-sync python gr00t/eval/run_gr00t_server.py \
      --model-path "${CHECKPOINT_PATH}" \
      --embodiment-tag LIBERO_PANDA \
      --device cuda:0 \
      --host 0.0.0.0 \
      --port "${POLICY_PORT}" \
      --seed "${BASE_SEED}" \
      --use-sim-policy-wrapper
  ) >"${server_log}" 2>&1 &
  SERVER_PID=$!

  local server_ready=false
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "${EXPERIMENT_ID} policy server exited during startup. See ${server_log}." >&2
      exit 1
    fi
    if PYTHONPATH="${PROJECT_ROOT}/src" "${PROJECT_ROOT}/.venv-a0/bin/python" - <<'PY' \
      >/dev/null 2>&1
from unitree_gr00t.a0 import RemotePolicyClient

with RemotePolicyClient("127.0.0.1", 5550, timeout_ms=1000) as client:
    raise SystemExit(0 if client.ping() else 1)
PY
    then
      server_ready=true
      break
    fi
    sleep 2
  done
  if [[ "${server_ready}" != true ]]; then
    echo "${EXPERIMENT_ID} policy server did not become ready. See ${server_log}." >&2
    exit 1
  fi
}

run_shard() {
  local horizon="$1"
  local name="$2"
  local task_types_csv="$3"
  local cases_csv="${4:-}"
  local task_types=()
  local cases=()
  IFS=',' read -r -a task_types <<<"${task_types_csv}"
  if [[ -n "${cases_csv}" ]]; then
    IFS=',' read -r -a cases <<<"${cases_csv}"
  fi
  local run_dir="${SHARD_ROOT}/H${horizon}/${name}"
  mkdir -p "${run_dir}"
  local command=(
    /usr/bin/env "PYTHONPATH=${PROJECT_ROOT}/src"
    "${PROJECT_ROOT}/.venv-a0/bin/python" -m unitree_gr00t.a0_eval
    --robocerebra-source "${PROJECT_ROOT}/.upstream/RoboCerebra"
    --benchmark-dir "${PROJECT_ROOT}/.cache/robocerebra/bench"
    --checkpoint "${CHECKPOINT_PATH}"
    --benchmark-revision 2573426c13dfcd5e7d7831c15587b058aaa1c0c0
    --model-revision "${MODEL_REVISION}"
    --dataset-revision 4e386b9aa266f05b199739d7b58950252244ea21
    --task-types "${task_types[@]}"
    --experiment-id "${EXPERIMENT_ID}"
    --variant "${VARIANT}"
    --trials "${TRIALS}"
    --execution-horizon "${horizon}"
    --control-frequency-hz 20
    --steps-per-subtask 150
    --initial-wait-steps 15
    --post-success-steps 80
    --seed "${BASE_SEED}"
    --policy-host 127.0.0.1
    --policy-port "${POLICY_PORT}"
    --output "${run_dir}"
    --resume
  )
  if [[ "${#cases[@]}" -gt 0 ]]; then
    command+=(--cases "${cases[@]}")
  fi
  "${command[@]}" >>"${run_dir}/run.log" 2>&1
}

run_parallel_h16() {
  echo "Starting five H16 simulator workers for the 407 remaining unique episodes"
  run_shard 16 memory-execution-tail Memory_Execution case10 &
  local pids=("$!")
  run_shard 16 memory-exploration Memory_Exploration & pids+=("$!")
  run_shard 16 mix Mix & pids+=("$!")
  run_shard 16 observation-mismatching Observation_Mismatching & pids+=("$!")
  run_shard 16 random-disturbance Random_Disturbance & pids+=("$!")
  local failed=0
  for worker_pid in "${pids[@]}"; do
    wait "${worker_pid}" || failed=1
  done
  [[ "${failed}" == 0 ]] || { echo "At least one H16 shard failed" >&2; return 1; }
}

run_parallel_h8() {
  echo "Starting eight call-balanced H8 simulator workers for all 600 episodes"
  run_shard 8 memory-execution-head Memory_Execution case1,case2,case3,case4,case5 &
  local pids=("$!")
  run_shard 8 memory-execution-tail Memory_Execution case6,case7,case8,case9,case10 & pids+=("$!")
  run_shard 8 memory-exploration-head Memory_Exploration case1,case2,case3,case4,case5,case6 & pids+=("$!")
  run_shard 8 mix-head Mix case1,case2,case3,case4,case5,case6 & pids+=("$!")
  run_shard 8 memory-exploration-ideal-tail Memory_Exploration,Ideal case7,case8,case9,case10 & pids+=("$!")
  run_shard 8 mix-observation-tail Mix,Observation_Mismatching case7,case8,case9,case10 & pids+=("$!")
  run_shard 8 ideal-observation-head Ideal,Observation_Mismatching case1,case2,case3,case4,case5,case6 & pids+=("$!")
  run_shard 8 random-disturbance Random_Disturbance & pids+=("$!")
  local failed=0
  for worker_pid in "${pids[@]}"; do
    wait "${worker_pid}" || failed=1
  done
  [[ "${failed}" == 0 ]] || { echo "At least one H8 shard failed" >&2; return 1; }
}

if [[ "${ROBOCEREBRA_FUNCTIONS_ONLY:-${A0_FUNCTIONS_ONLY:-false}}" == true ]]; then
  return 0 2>/dev/null || exit 0
fi

if ! jq -e '.complete == true and .episodes == 600' \
  "${ARTIFACT_ROOT}/H16/summary.json" >/dev/null 2>&1; then
  start_server H16-parallel
  run_parallel_h16
  stop_server

  PYTHONPATH="${PROJECT_ROOT}/src" python3 -m unitree_gr00t.a0_merge \
    --target "${ARTIFACT_ROOT}/H16" \
    --shard "${SHARD_ROOT}/H16/memory-execution-tail" \
    --shard "${SHARD_ROOT}/H16/memory-exploration" \
    --shard "${SHARD_ROOT}/H16/mix" \
    --shard "${SHARD_ROOT}/H16/observation-mismatching" \
    --shard "${SHARD_ROOT}/H16/random-disturbance"
else
  echo "H16 target already has 600 validated episodes; skipping it"
fi

if ! jq -e '.complete == true and .episodes == 600' \
  "${ARTIFACT_ROOT}/H8/summary.json" >/dev/null 2>&1; then
  start_server H8-parallel
  run_parallel_h8
  stop_server

  PYTHONPATH="${PROJECT_ROOT}/src" python3 -m unitree_gr00t.a0_merge \
    --create-target \
    --target "${ARTIFACT_ROOT}/H8" \
    --shard "${SHARD_ROOT}/H8/memory-execution-head" \
    --shard "${SHARD_ROOT}/H8/memory-execution-tail" \
    --shard "${SHARD_ROOT}/H8/memory-exploration-head" \
    --shard "${SHARD_ROOT}/H8/mix-head" \
    --shard "${SHARD_ROOT}/H8/memory-exploration-ideal-tail" \
    --shard "${SHARD_ROOT}/H8/mix-observation-tail" \
    --shard "${SHARD_ROOT}/H8/ideal-observation-head" \
    --shard "${SHARD_ROOT}/H8/random-disturbance"
else
  echo "H8 target already has 600 validated episodes; skipping it"
fi

PYTHONPATH="${PROJECT_ROOT}/src" "${PROJECT_ROOT}/.venv-a0/bin/python" \
  -m unitree_gr00t.a0_report \
  --run "H16=${ARTIFACT_ROOT}/H16" \
  --run "H8=${ARTIFACT_ROOT}/H8" \
  --output "${ARTIFACT_ROOT}/reviewer" \
  --experiment-id "${EXPERIMENT_ID}" \
  --variant "${VARIANT}" \
  >"${ARTIFACT_ROOT}/reviewer-report.log" 2>&1

echo "${EXPERIMENT_ID} full benchmark and reviewer report completed: ${ARTIFACT_ROOT}"
