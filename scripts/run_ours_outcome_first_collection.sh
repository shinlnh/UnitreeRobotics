#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
PORT="${PORT:-5551}"
MAX_STATES_PER_EPISODE="${MAX_STATES_PER_EPISODE:-8}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-150}"
MAX_POLICY_CALLS="${MAX_POLICY_CALLS:-48}"
CONSENSUS_HYPOTHESES="${CONSENSUS_HYPOTHESES:-4}"
RETURN_TARGET="outcome-first-physical-v1"
COLLECTION_ID="R0-residual-v9-outcome-first"

if pgrep -f 'python .*unitree_gr00t\.(ours_server|b_server)' >/dev/null; then
  echo "Refusing to load a second GR00T server while another research server is active" >&2
  exit 1
fi
if ss -ltn | grep -q ":${PORT} "; then
  echo "Refusing occupied outcome-first policy port ${PORT}" >&2
  exit 1
fi

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

server_log="artifacts/Ours/counterfactual/${COLLECTION_ID}-server.log"
mkdir -p "$(dirname "${server_log}")"
PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.b_server \
  --checkpoint checkpoints/robocerebra/GR00T-RC \
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
  --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
  --embodiment LIBERO_PANDA \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --seed 10007 \
  >"${server_log}" 2>&1 &
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
  echo "Outcome-first B server did not open port ${PORT}" >&2
  exit 1
fi

for base_seed in 10007 11007 12007; do
  source_rollout="outputs/robocerebra/ctr-residual-rollouts-v1/train-seed${base_seed}-B-retry-H16"
  source_corpus="outputs/robocerebra/ctr-residual-live-corpus-v1/train-seed${base_seed}-B-retry-H16"
  destination="outputs/robocerebra/ctr-counterfactual-rollout-v9-outcome-first-seed${base_seed}"
  artifact_root="artifacts/Ours/counterfactual/${COLLECTION_ID}-seed${base_seed}"
  mkdir -p "${artifact_root}"
  if [[ -f "${destination}/manifest.json" ]]; then
    PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_audit \
      --corpus "${destination}" \
      --output "${artifact_root}/corpus_audit.json" \
      >"${artifact_root}/corpus_audit.log"
    continue
  fi
  if [[ -d "${destination}" ]] && [[ -n "$(find "${destination}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing incomplete outcome-first corpus: ${destination}" >&2
    exit 1
  fi
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_counterfactual_rollout_prepare \
    --rollout "${source_rollout}" \
    --source-corpus "${source_corpus}" \
    --robocerebra-source .upstream/RoboCerebra \
    --benchmark-dir .cache/robocerebra/bench \
    --destination "${destination}" \
    --policy-host 127.0.0.1 \
    --policy-port "${PORT}" \
    --stop-stride 1 \
    --max-states-per-episode "${MAX_STATES_PER_EPISODE}" \
    --min-source-elapsed-steps 75 \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --max-policy-calls "${MAX_POLICY_CALLS}" \
    --consensus-hypotheses "${CONSENSUS_HYPOTHESES}" \
    --return-target "${RETURN_TARGET}" \
    --require-stop-pending \
    --residual-retry-baseline \
    >"${artifact_root}/generator.log" 2>&1
  PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_counterfactual_audit \
    --corpus "${destination}" \
    --output "${artifact_root}/corpus_audit.json" \
    >"${artifact_root}/corpus_audit.log"
done

stop_server
