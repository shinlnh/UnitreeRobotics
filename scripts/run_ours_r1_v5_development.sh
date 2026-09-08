#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/Ours/development/R1-v5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-counterfactual-v5-multiseed}"
OPTION_REGISTRY="${OPTION_REGISTRY:-artifacts/Ours/search/R0-counterfactual-v5-multiseed/option_registry.json}"
BASELINE="${BASELINE:-artifacts/Ours/development/control/B-retry-seed20007-H16}"
PORT="${PORT:-5550}"
mkdir -p "${OUTPUT_ROOT}/servers"

mapfile -t ranked < <(PYTHONPATH=src .venv/bin/python - "${OPTION_REGISTRY}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("complete") or len(payload.get("variants", [])) != 6:
    raise SystemExit("R0 option registry is incomplete")
for result in sorted(payload["variants"], key=lambda row: row["offline_option_rank"])[:4]:
    margin = result["option_validation"]["selective_recovery"]["option_value_margin"]
    print(result["variant_id"], format(float(margin), ".9g"))
PY
)
if [[ "${#ranked[@]}" -ne 4 ]]; then
  echo "R1 requires the top four completed R0 variants" >&2
  exit 1
fi
read -r rank1 margin1 <<<"${ranked[0]}"
read -r rank2 margin2 <<<"${ranked[1]}"
read -r rank3 margin3 <<<"${ranked[2]}"
read -r rank4 margin4 <<<"${ranked[3]}"
safer_margin1=$(PYTHONPATH=src .venv/bin/python - "${margin1}" <<'PY'
import sys
print(format(float(sys.argv[1]) + 0.05, ".9g"))
PY
)

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
  echo "Ours R1 server did not open port ${PORT}" >&2
  exit 1
}

run_variant() {
  local variant_id="$1"
  local checkpoint_name="$2"
  local hypotheses="$3"
  local option_margin="$4"
  local checkpoint="${CHECKPOINT_ROOT}/${checkpoint_name}"
  local output="${OUTPUT_ROOT}/${variant_id}"
  if [[ -f "${output}/summary.json" ]] && \
    "${EVAL_PYTHON}" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' "${output}/summary.json"; then
    return
  fi
  local resume=()
  [[ -f "${output}/run_manifest.json" ]] && resume=(--resume)
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
    --gate-signal completion \
    --consensus-hypotheses "${hypotheses}" \
    --consensus-cooldown-decisions 16 \
    --max-recovery-attempts 1 \
    --min-recovery-elapsed-steps 75 \
    --use-option-values \
    --option-value-margin "${option_margin}" \
    "${resume[@]}" \
    >"${OUTPUT_ROOT}/${variant_id}.log" 2>&1
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_rollout_audit \
    --rollout "${output}" \
    --output "${output}/belief_audit.json" \
    >>"${OUTPUT_ROOT}/${variant_id}.log" 2>&1
}

# Seven slots remain after registered r1-00..r1-04. The identities below are
# fixed before R0 v5 results; rank labels refer only to the train-seed audit.
# id checkpoint hypotheses option-margin
variants=(
  "r1-05-v5-rank1-c1 ${rank1} 1 ${margin1}"
  "r1-06-v5-rank1-c4 ${rank1} 4 ${margin1}"
  "r1-07-v5-rank1-c1-safe ${rank1} 1 ${safer_margin1}"
  "r1-08-v5-rank2-c1 ${rank2} 1 ${margin2}"
  "r1-09-v5-rank2-c4 ${rank2} 4 ${margin2}"
  "r1-10-v5-rank3-c1 ${rank3} 1 ${margin3}"
  "r1-11-v5-rank4-c1 ${rank4} 1 ${margin4}"
)

current_checkpoint=""
for specification in "${variants[@]}"; do
  read -r variant_id checkpoint_name hypotheses option_margin <<<"${specification}"
  if [[ "${checkpoint_name}" != "${current_checkpoint}" ]]; then
    stop_server
    start_server "${CHECKPOINT_ROOT}/${checkpoint_name}" "${checkpoint_name}"
    current_checkpoint="${checkpoint_name}"
  fi
  run_variant "${variant_id}" "${checkpoint_name}" "${hypotheses}" "${option_margin}"
done
stop_server

PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_development_search \
  --baseline "${BASELINE}" \
  --runs "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/registry.json" \
  --expected-variants 7 \
  --bootstrap-resamples 10000 \
  --seed 20007 \
  >"${OUTPUT_ROOT}/registry.log" 2>&1
