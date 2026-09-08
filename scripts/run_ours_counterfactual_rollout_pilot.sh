#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
PORT="${PORT:-5551}"
DESTINATION="${DESTINATION:-outputs/robocerebra/ctr-counterfactual-rollout-v2-pilot}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/Ours/counterfactual/R0-rollforward-pilot}"
STOP_STRIDE="${STOP_STRIDE:-128}"
MAX_STATES_PER_EPISODE="${MAX_STATES_PER_EPISODE:-1}"
MIN_SOURCE_ELAPSED_STEPS="${MIN_SOURCE_ELAPSED_STEPS:-75}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-75}"
MAX_POLICY_CALLS="${MAX_POLICY_CALLS:-24}"
CONSENSUS_HYPOTHESES="${CONSENSUS_HYPOTHESES:-4}"
mkdir -p "${ARTIFACT_ROOT}"

if pgrep -f 'python .*unitree_gr00t\.(ours_server|b_server)' >/dev/null; then
  echo "Refusing to load a second GR00T server while another research server is active" >&2
  exit 1
fi
if ss -ltn | grep -q ":${PORT} "; then
  echo "Refusing occupied counterfactual policy port ${PORT}" >&2
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

PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.b_server \
  --checkpoint checkpoints/robocerebra/GR00T-RC \
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
  --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
  --embodiment LIBERO_PANDA \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --seed 10007 \
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
  echo "Counterfactual B server did not open port ${PORT}" >&2
  exit 1
fi

PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_counterfactual_rollout_prepare \
  --rollout outputs/robocerebra/ctr-live-rollouts-v1/train-seed10007-full-c1-H16 \
  --source-corpus outputs/robocerebra/ctr-live-corpus-v2/train-seed10007-full-c1-H16 \
  --robocerebra-source .upstream/RoboCerebra \
  --benchmark-dir .cache/robocerebra/bench \
  --destination "${DESTINATION}" \
  --policy-host 127.0.0.1 \
  --policy-port "${PORT}" \
  --stop-stride "${STOP_STRIDE}" \
  --max-states-per-episode "${MAX_STATES_PER_EPISODE}" \
  --min-source-elapsed-steps "${MIN_SOURCE_ELAPSED_STEPS}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --max-policy-calls "${MAX_POLICY_CALLS}" \
  --consensus-hypotheses "${CONSENSUS_HYPOTHESES}" \
  >"${ARTIFACT_ROOT}/generator.log" 2>&1

PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_audit \
  --corpus "${DESTINATION}" \
  --output "${ARTIFACT_ROOT}/pilot_audit.json" \
  | tee "${ARTIFACT_ROOT}/pilot_audit.log"
