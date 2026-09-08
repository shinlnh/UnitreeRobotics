#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/Ours/development/R1b-residual-v7}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-residual-v7-confirmed}"
OPTION_REGISTRY="${OPTION_REGISTRY:-artifacts/Ours/search/R0-residual-v7-confirmed/residual_registry.json}"
BASELINE="${BASELINE:-artifacts/Ours/development/control/B-retry-seed22007-H16}"
PORT="${PORT:-5550}"
BASE_SEED="${BASE_SEED:-22007}"
mkdir -p "${OUTPUT_ROOT}/servers" "$(dirname "${BASELINE}")"

if [[ "${BASE_SEED}" == "7" ]]; then
  echo "R1b cannot consume the final held-out seed" >&2
  exit 1
fi

mapfile -t ranked < <(PYTHONPATH=src .venv/bin/python - "${OPTION_REGISTRY}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    not payload.get("complete")
    or not payload.get("residual_retry_baseline")
    or len(payload.get("variants", [])) != 6
):
    raise SystemExit("R0 residual registry is incomplete")
for result in sorted(payload["variants"], key=lambda row: row["offline_option_rank"])[:4]:
    margin = result["option_validation"]["selective_recovery"]["option_value_margin"]
    print(result["variant_id"], format(float(margin), ".9g"))
PY
)
if [[ "${#ranked[@]}" -ne 4 ]]; then
  echo "R1b requires the top four completed residual models" >&2
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

wait_for_server() {
  # Allow a cold CUDA context enough time to load the frozen 3B VLA and the
  # largest registered recovery head.
  for _ in $(seq 1 240); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      wait "${server_pid}"
    fi
    if ss -ltn | grep -q ":${PORT} "; then
      return
    fi
    sleep 1
  done
  echo "Development server did not open port ${PORT}" >&2
  exit 1
}

guard_port() {
  if ss -ltn | grep -q ":${PORT} "; then
    echo "Refusing occupied Ours development port ${PORT}" >&2
    exit 1
  fi
}

start_b_server() {
  guard_port
  PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.b_server \
    --checkpoint checkpoints/robocerebra/GR00T-RC \
    --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
    --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
    --embodiment LIBERO_PANDA \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --seed "${BASE_SEED}" \
    >"${OUTPUT_ROOT}/servers/B-retry-seed${BASE_SEED}.log" 2>&1 &
  server_pid="$!"
  wait_for_server
}

start_ours_server() {
  local checkpoint="$1"
  local server_id="$2"
  guard_port
  PYTHONPATH=src "${SERVER_PYTHON}" -m unitree_gr00t.ours_server \
    --checkpoint checkpoints/robocerebra/GR00T-RC \
    --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
    --recovery-checkpoint "${checkpoint}" \
    --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
    --embodiment LIBERO_PANDA \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --seed "${BASE_SEED}" \
    >"${OUTPUT_ROOT}/servers/${server_id}.log" 2>&1 &
  server_pid="$!"
  wait_for_server
}

common_eval_args=(
  --robocerebra-source .upstream/RoboCerebra
  --benchmark-dir .cache/robocerebra/bench
  --checkpoint checkpoints/robocerebra/GR00T-RC
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector
  --benchmark-revision 2573426c13dfcd5e7d7831c15587b058aaa1c0c0
  --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96
  --dataset-revision 4e386b9aa266f05b199739d7b58950252244ea21
  --planner RoboCerebra-HPE-fixed-anchor-reimplementation
  --plan-source "canonical task_description.txt step annotations"
  --task-types Ideal Memory_Execution Memory_Exploration Mix Observation_Mismatching Random_Disturbance
  --trials 1
  --execution-horizon 16
  --stop-confirmation-window 2
  --control-frequency-hz 20
  --steps-per-subtask 150
  --initial-wait-steps 15
  --post-success-steps 80
  --seed "${BASE_SEED}"
  --policy-host 127.0.0.1
  --policy-port "${PORT}"
  --no-trace-images
)

if [[ ! -f "${BASELINE}/summary.json" ]] || \
  ! "${EVAL_PYTHON}" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' "${BASELINE}/summary.json"; then
  start_b_server
  baseline_resume=()
  [[ -f "${BASELINE}/run_manifest.json" ]] && baseline_resume=(--resume)
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.b_retry_eval \
    "${common_eval_args[@]}" \
    --output "${BASELINE}" \
    "${baseline_resume[@]}" \
    >"${OUTPUT_ROOT}/B-retry-seed${BASE_SEED}.log" 2>&1
  stop_server
fi

run_variant() {
  local variant_id="$1"
  local checkpoint_name="$2"
  local hypotheses="$3"
  local option_margin="$4"
  local failure_mode="$5"
  local checkpoint="${CHECKPOINT_ROOT}/${checkpoint_name}"
  local output="${OUTPUT_ROOT}/${variant_id}"
  if [[ -f "${output}/summary.json" ]] && \
    "${EVAL_PYTHON}" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1]))["complete"])' "${output}/summary.json"; then
    return
  fi
  local resume=()
  local failure_args=()
  [[ -f "${output}/run_manifest.json" ]] && resume=(--resume)
  [[ "${failure_mode}" == "option-only" ]] && failure_args=(--failure-threshold 0)
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_eval \
    "${common_eval_args[@]}" \
    --recovery-checkpoint "${checkpoint}" \
    --output "${output}" \
    --gate-signal completion \
    --consensus-hypotheses "${hypotheses}" \
    --consensus-cooldown-decisions 16 \
    --max-recovery-attempts 1 \
    --min-recovery-elapsed-steps 75 \
    --use-option-values \
    --option-value-margin "${option_margin}" \
    --residual-retry-baseline \
    "${failure_args[@]}" \
    "${resume[@]}" \
    >"${OUTPUT_ROOT}/${variant_id}.log" 2>&1
  PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_rollout_audit \
    --rollout "${output}" \
    --output "${output}/belief_audit.json" \
    >>"${OUTPUT_ROOT}/${variant_id}.log" 2>&1
}

# Frozen before seed-22007 is consumed.  Counterfactual CONSENSUS_PREFIX labels
# use four hypotheses, so C4 is the matched primary setting and C1 is an
# explicit ablation.  The option-only variants trust the held-out residual
# margin; the checkpoint failure gate is tested separately for rank 1.
# id checkpoint hypotheses margin failure-gate
variants=(
  "r1b-00-rank1-option-c4 ${rank1} 4 ${margin1} option-only"
  "r1b-01-rank2-option-c4 ${rank2} 4 ${margin2} option-only"
  "r1b-02-rank3-option-c4 ${rank3} 4 ${margin3} option-only"
  "r1b-03-rank4-option-c4 ${rank4} 4 ${margin4} option-only"
  "r1b-04-rank1-gated-c4 ${rank1} 4 ${margin1} checkpoint"
  "r1b-05-rank1-option-c1 ${rank1} 1 ${margin1} option-only"
  "r1b-06-rank1-safe-c4 ${rank1} 4 ${safer_margin1} option-only"
)

current_checkpoint=""
for specification in "${variants[@]}"; do
  read -r variant_id checkpoint_name hypotheses option_margin failure_mode <<<"${specification}"
  if [[ "${checkpoint_name}" != "${current_checkpoint}" ]]; then
    stop_server
    start_ours_server "${CHECKPOINT_ROOT}/${checkpoint_name}" "${checkpoint_name}"
    current_checkpoint="${checkpoint_name}"
  fi
  run_variant "${variant_id}" "${checkpoint_name}" "${hypotheses}" "${option_margin}" "${failure_mode}"
done
stop_server

PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_development_search \
  --baseline "${BASELINE}" \
  --runs "${OUTPUT_ROOT}" \
  --run-prefix r1b- \
  --output "${OUTPUT_ROOT}/registry.json" \
  --expected-variants 7 \
  --bootstrap-resamples 10000 \
  --seed "${BASE_SEED}" \
  >"${OUTPUT_ROOT}/registry.log" 2>&1
