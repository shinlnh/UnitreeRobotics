#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/Ours/development/R1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-live-v2}"
PORT="${PORT:-5550}"
mkdir -p "${OUTPUT_ROOT}/servers"

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

start_server() {
  local checkpoint="$1"
  local server_id="$2"
  if ss -ltn | grep -q ":${PORT} "; then
    echo "Refusing occupied Ours development port ${PORT}" >&2
    exit 1
  fi
  PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.ours_server \
    --checkpoint checkpoints/robocerebra/GR00T-RC \
    --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
    --recovery-checkpoint "${checkpoint}" \
    --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
    --embodiment LIBERO_PANDA \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --seed 20007 \
    >"${OUTPUT_ROOT}/servers/${server_id}.log" 2>&1 &
  server_pid="$!"
  for _ in $(seq 1 90); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      wait "${server_pid}"
    fi
    if ss -ltn | grep -q ":${PORT} "; then
      return
    fi
    sleep 1
  done
  echo "Ours development server did not open port ${PORT}" >&2
  exit 1
}

run_variant() {
  local variant_id="$1"
  local checkpoint_name="$2"
  local gate_signal="$3"
  local hypotheses="$4"
  local failure_threshold="$5"
  local cooldown="$6"
  local boundary="$7"
  local checkpoint="${CHECKPOINT_ROOT}/${checkpoint_name}"
  local output="${OUTPUT_ROOT}/${variant_id}"
  if [[ -f "${output}/summary.json" ]] && \
    "${EVAL_PYTHON}" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' "${output}/summary.json"; then
    return
  fi
  local resume=()
  [[ -f "${output}/run_manifest.json" ]] && resume=(--resume)
  local boundary_args=()
  [[ "${boundary}" != "none" ]] && boundary_args=(--stagnation-boundary-steps "${boundary}")
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_eval \
    --robocerebra-source .upstream/RoboCerebra \
    --benchmark-dir .cache/robocerebra/bench \
    --checkpoint checkpoints/robocerebra/GR00T-RC \
    --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
    --recovery-checkpoint "${checkpoint}" \
    --benchmark-revision 2573426c13dfcd5e7d7831c15587b058aaa1c0c0 \
    --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
    --dataset-revision 4e386b9aa266f05b199739d7b58950252244ea21 \
    --planner RoboCerebra-HPE-fixed-anchor-reimplementation \
    --plan-source "canonical task_description.txt step annotations" \
    --task-types Ideal Memory_Execution Memory_Exploration Mix Observation_Mismatching Random_Disturbance \
    --trials 1 \
    --execution-horizon 16 \
    --stop-confirmation-window 2 \
    --control-frequency-hz 20 \
    --steps-per-subtask 150 \
    --initial-wait-steps 15 \
    --post-success-steps 80 \
    --seed 20007 \
    --policy-host 127.0.0.1 \
    --policy-port "${PORT}" \
    --output "${output}" \
    --no-trace-images \
    --gate-signal "${gate_signal}" \
    --consensus-hypotheses "${hypotheses}" \
    --failure-threshold "${failure_threshold}" \
    --consensus-cooldown-decisions "${cooldown}" \
    "${boundary_args[@]}" \
    "${resume[@]}" \
    >"${OUTPUT_ROOT}/${variant_id}.log" 2>&1
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_rollout_audit \
    --rollout "${output}" \
    --output "${output}/belief_audit.json" \
    >>"${OUTPUT_ROOT}/${variant_id}.log" 2>&1
}

# id checkpoint gate hypotheses failure-threshold cooldown boundary
variants=(
  "r1-00-mlp025-comp-c1-b150 v01-mlp-h16-live0250 completion 1 0.90 16 150"
  "r1-01-mlp025-comp-c4-f090-c16-b150 v01-mlp-h16-live0250 completion 4 0.90 16 150"
  "r1-02-mlp025-comp-c4-f095-c16-b150 v01-mlp-h16-live0250 completion 4 0.95 16 150"
  "r1-03-mlp025-comp-c4-f090-c32-b150 v01-mlp-h16-live0250 completion 4 0.90 32 150"
  "r1-04-mlp025-comp-c1-b100 v01-mlp-h16-live0250 completion 1 0.90 16 100"
  "r1-05-mlp025-comp-c1-b225 v01-mlp-h16-live0250 completion 1 0.90 16 225"
  "r1-06-mlp025-comp-c1-none v01-mlp-h16-live0250 completion 1 0.90 16 none"
  "r1-07-mlp025-progress-c1-b150 v01-mlp-h16-live0250 progress 1 0.90 16 150"
  "r1-08-mlp025-maximum-c1-b150 v01-mlp-h16-live0250 maximum 1 0.90 16 150"
  "r1-09-mlp0125-comp-c1-b150 v00-mlp-h16-live0125 completion 1 0.90 16 150"
  "r1-10-mlp0500-comp-c1-b150 v02-mlp-h16-live0500 completion 1 0.90 16 150"
  "r1-11-gru0500-comp-c1-b150 v06-gru-h16-live0500 completion 1 0.90 16 150"
)

current_checkpoint=""
for specification in "${variants[@]}"; do
  read -r variant_id checkpoint_name gate_signal hypotheses failure_threshold cooldown boundary <<<"${specification}"
  if [[ "${checkpoint_name}" != "${current_checkpoint}" ]]; then
    stop_server
    start_server "${CHECKPOINT_ROOT}/${checkpoint_name}" "${checkpoint_name}"
    current_checkpoint="${checkpoint_name}"
  fi
  run_variant \
    "${variant_id}" "${checkpoint_name}" "${gate_signal}" "${hypotheses}" \
    "${failure_threshold}" "${cooldown}" "${boundary}"
done
