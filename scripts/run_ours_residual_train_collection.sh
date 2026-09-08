#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
RECOVERY_CHECKPOINT="${RECOVERY_CHECKPOINT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-counterfactual-v5-multiseed/v01-linear-h8-w64-d025}"
PORT="${PORT:-5550}"
COLLECTION_ROOT="${COLLECTION_ROOT:-artifacts/Ours/collection/residual-source-v1}"
mkdir -p "${COLLECTION_ROOT}/servers"

if pgrep -f 'python .*unitree_gr00t\.(ours_server|b_server)' >/dev/null; then
  echo "Refusing to load a second GR00T server while another research server is active" >&2
  exit 1
fi
if ss -ltn | grep -q ":${PORT} "; then
  echo "Refusing occupied residual-source policy port ${PORT}" >&2
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

PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.ours_server \
  --checkpoint checkpoints/robocerebra/GR00T-RC \
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
  --recovery-checkpoint "${RECOVERY_CHECKPOINT}" \
  --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
  --embodiment LIBERO_PANDA \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --seed 10007 \
  >"${COLLECTION_ROOT}/servers/ours.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 240); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  if ss -ltn | grep -q ":${PORT} "; then
    break
  fi
  sleep 1
done
if ! ss -ltn | grep -q ":${PORT} "; then
  echo "Residual-source server did not open port ${PORT}" >&2
  exit 1
fi

for base_seed in 10007 11007 12007; do
  rollout="outputs/robocerebra/ctr-residual-rollouts-v1/train-seed${base_seed}-B-retry-H16"
  corpus="outputs/robocerebra/ctr-residual-live-corpus-v1/train-seed${base_seed}-B-retry-H16"
  artifact="${COLLECTION_ROOT}/train-seed${base_seed}-B-retry-H16"
  mkdir -p "${artifact}"
  if [[ ! -f "${rollout}/summary.json" ]] || \
    ! "${EVAL_PYTHON}" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' "${rollout}/summary.json"; then
    resume=()
    [[ -f "${rollout}/run_manifest.json" ]] && resume=(--resume)
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
      --seed "${base_seed}" \
      --policy-host 127.0.0.1 \
      --policy-port "${PORT}" \
      --output "${rollout}" \
      --no-trace-images \
      --gate-signal completion \
      --consensus-hypotheses 4 \
      --max-recovery-attempts 1 \
      --min-recovery-elapsed-steps 75 \
      --use-option-values \
      --option-value-margin 1000000 \
      --failure-threshold 0 \
      --residual-retry-baseline \
      --capture-training-context \
      "${resume[@]}" \
      >"${artifact}/rollout.log" 2>&1
  fi

  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_rollout_audit \
    --rollout "${rollout}" \
    --output "${artifact}/rollout_audit.json" \
    >"${artifact}/rollout_audit.log"
  PYTHONPATH=src "${EVAL_PYTHON}" - "${rollout}" <<'PY'
import json
import sys
from pathlib import Path

from unitree_gr00t.ours_counterfactual_rollout_prepare import _validate_residual_source

rollout = Path(sys.argv[1])
_validate_residual_source(
    json.loads((rollout / "run_manifest.json").read_text(encoding="utf-8")),
    json.loads((rollout / "summary.json").read_text(encoding="utf-8")),
)
PY
  if [[ ! -f "${corpus}/manifest.json" ]]; then
    if [[ -d "${corpus}" ]] && [[ -n "$(find "${corpus}" -mindepth 1 -print -quit)" ]]; then
      echo "Refusing incomplete residual source corpus: ${corpus}" >&2
      exit 1
    fi
    PYTHONPATH=src .venv/bin/python -m unitree_gr00t.ours_rollout_prepare \
      --rollout "${rollout}" \
      --destination "${corpus}" \
      >"${artifact}/corpus_prepare.log"
  fi
done

stop_server
