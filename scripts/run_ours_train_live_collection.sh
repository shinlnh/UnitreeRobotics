#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
BASE_SEED="${BASE_SEED:-11007}"
PORT="${PORT:-5550}"
RECOVERY_CHECKPOINT="${RECOVERY_CHECKPOINT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-calibrated/v03-mlp-h16-w128-l1}"
ROLLOUT="${ROLLOUT:-outputs/robocerebra/ctr-live-rollouts-v1/train-seed${BASE_SEED}-full-c1-H16}"
CORPUS="${CORPUS:-outputs/robocerebra/ctr-live-corpus-v2/train-seed${BASE_SEED}-full-c1-H16}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/Ours/collection/train-seed${BASE_SEED}-full-c1-H16}"

case "${BASE_SEED}" in
  10007|11007|12007) ;;
  *)
    echo "Training-context collection is restricted to seeds 10007, 11007, and 12007" >&2
    exit 1
    ;;
esac

mkdir -p "${ARTIFACT_ROOT}"
if pgrep -f 'python .*unitree_gr00t\.(ours_server|b_server)' >/dev/null; then
  echo "Refusing to load a second GR00T server while another research server is active" >&2
  exit 1
fi
if ss -ltn | grep -q ":${PORT} "; then
  echo "Refusing occupied training-collection policy port ${PORT}" >&2
  exit 1
fi

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
}
trap stop_server EXIT INT TERM

PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.ours_server \
  --checkpoint checkpoints/robocerebra/GR00T-RC \
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
  --recovery-checkpoint "${RECOVERY_CHECKPOINT}" \
  --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
  --embodiment LIBERO_PANDA \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --seed "${BASE_SEED}" \
  >"${ARTIFACT_ROOT}/server.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 90); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  if ss -ltn | grep -q ":${PORT} "; then
    break
  fi
  sleep 1
done
if ! ss -ltn | grep -q ":${PORT} "; then
  echo "Training-collection server did not open port ${PORT}" >&2
  exit 1
fi

resume=()
[[ -f "${ROLLOUT}/run_manifest.json" ]] && resume=(--resume)
PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_eval \
  --robocerebra-source .upstream/RoboCerebra \
  --benchmark-dir .cache/robocerebra/bench \
  --checkpoint checkpoints/robocerebra/GR00T-RC \
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
  --recovery-checkpoint "${RECOVERY_CHECKPOINT}" \
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
  --seed "${BASE_SEED}" \
  --policy-host 127.0.0.1 \
  --policy-port "${PORT}" \
  --output "${ROLLOUT}" \
  --no-trace-images \
  --gate-signal progress \
  --consensus-hypotheses 1 \
  --max-recovery-attempts 1 \
  --min-recovery-elapsed-steps 75 \
  --capture-training-context \
  --collection-force-boundary-steps 150 \
  "${resume[@]}" \
  >"${ARTIFACT_ROOT}/rollout.log" 2>&1

stop_server
server_pid=""

if [[ -d "${CORPUS}" ]]; then
  echo "Refusing existing live corpus destination: ${CORPUS}" >&2
  exit 1
fi
PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_rollout_audit \
  --rollout "${ROLLOUT}" \
  --output "${ARTIFACT_ROOT}/rollout_audit.json" \
  >"${ARTIFACT_ROOT}/rollout_audit.log"
PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_rollout_prepare \
  --rollout "${ROLLOUT}" \
  --destination "${CORPUS}" \
  >"${ARTIFACT_ROOT}/corpus_prepare.log"
