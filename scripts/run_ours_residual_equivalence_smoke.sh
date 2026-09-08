#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${SERVER_PYTHON:-.upstream/Isaac-GR00T-N1.7/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-.venv-a0/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/Ours/smoke/residual-equivalence-cfffb2e-v2}"
RECOVERY_CHECKPOINT="${RECOVERY_CHECKPOINT:-checkpoints/robocerebra/GR00T-RC-CTR-search/R0-counterfactual-v5-multiseed/v01-linear-h8-w64-d025}"
PORT="${PORT:-5550}"
BASELINE="${OUTPUT_ROOT}/B-retry"
RESIDUAL="${OUTPUT_ROOT}/residual-abstain"
mkdir -p "${OUTPUT_ROOT}/servers"

if [[ -e "${BASELINE}" || -e "${RESIDUAL}" ]]; then
  echo "Refusing to overwrite residual-equivalence smoke artifacts" >&2
  exit 1
fi
if ss -ltn | grep -q ":${PORT} "; then
  echo "Refusing occupied smoke port ${PORT}" >&2
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

start_server() {
  local kind="$1"
  local module="unitree_gr00t.b_server"
  local extra=()
  if [[ "${kind}" == "ours" ]]; then
    module="unitree_gr00t.ours_server"
    extra=(--recovery-checkpoint "${RECOVERY_CHECKPOINT}")
  fi
  PYTHONPATH=src "${SERVER_PYTHON}" -m "${module}" \
    --checkpoint checkpoints/robocerebra/GR00T-RC \
    --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector \
    "${extra[@]}" \
    --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96 \
    --embodiment LIBERO_PANDA \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --seed 30007 \
    >"${OUTPUT_ROOT}/servers/${kind}.log" 2>&1 &
  server_pid="$!"
  for _ in $(seq 1 240); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      wait "${server_pid}"
    fi
    if ss -ltn | grep -q ":${PORT} "; then
      return
    fi
    sleep 1
  done
  echo "Residual-equivalence ${kind} server did not open port ${PORT}" >&2
  exit 1
}

common=(
  --robocerebra-source .upstream/RoboCerebra
  --benchmark-dir .cache/robocerebra/bench
  --checkpoint checkpoints/robocerebra/GR00T-RC
  --selector-checkpoint checkpoints/robocerebra/GR00T-RC-SparkVLA-selector
  --benchmark-revision 2573426c13dfcd5e7d7831c15587b058aaa1c0c0
  --model-revision 5d2e1e361bf65aabbe4d18179515f5a10936cc96
  --dataset-revision 4e386b9aa266f05b199739d7b58950252244ea21
  --planner RoboCerebra-HPE-fixed-anchor-reimplementation
  --plan-source "canonical task_description.txt step annotations"
  --task-types Ideal
  --cases case1
  --trials 1
  --execution-horizon 16
  --stop-confirmation-window 2
  --control-frequency-hz 20
  --steps-per-subtask 150
  --initial-wait-steps 15
  --post-success-steps 80
  --seed 30007
  --policy-host 127.0.0.1
  --policy-port "${PORT}"
  --no-trace-images
)

start_server b
PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.b_retry_eval \
  "${common[@]}" --output "${BASELINE}" >"${OUTPUT_ROOT}/B-retry.log" 2>&1
stop_server
start_server ours
PYTHONPATH=src "${EVAL_PYTHON}" -m unitree_gr00t.ours_eval \
  "${common[@]}" \
  --recovery-checkpoint "${RECOVERY_CHECKPOINT}" \
  --output "${RESIDUAL}" \
  --gate-signal completion \
  --consensus-hypotheses 4 \
  --max-recovery-attempts 1 \
  --min-recovery-elapsed-steps 75 \
  --use-option-values \
  --option-value-margin 1000000 \
  --failure-threshold 0 \
  --residual-retry-baseline \
  >"${OUTPUT_ROOT}/residual-abstain.log" 2>&1

PYTHONPATH=src .venv/bin/python - "${BASELINE}" "${RESIDUAL}" "${OUTPUT_ROOT}/equivalence.json" <<'PY'
import json
import sys
from pathlib import Path

baseline, residual, output = map(Path, sys.argv[1:])

def rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

decision_keys = (
    "task_type", "case", "trial", "policy_call", "step_before", "step_after",
    "decision_seed", "instruction", "full_task_instruction", "selector_candidate",
    "selector_scores", "selector_valid", "selector_context_sha256",
    "selector_anchor_history_sha256", "new_subgoal_anchor", "selected_prefix_length",
    "predicted_chunk", "prefix_selection", "stop_pending", "stop_committed",
    "stop_confirmation_streak_after", "active_subgoal_index_before",
    "active_subgoal_index_after", "completed_subtasks_before",
    "completed_subtasks_after", "success_before", "success_after",
    "success_predicates_before", "success_predicates_after", "transitions", "retry",
    "retry_trigger", "retry_triggered", "retry_attempt_index_before",
    "retry_attempt_index_after", "subgoal_advanced", "injections",
)
episode_keys = (
    "task_type", "case", "trial", "seed", "steps", "total_simulator_steps",
    "policy_calls", "predicted_actions", "selector_executed_actions",
    "selector_stop_proposals", "selector_stop_commits", "selector_subgoals_completed",
    "selector_prefix_histogram", "agent_completed_subtasks", "possible_subtasks",
    "final_success", "reached_success", "first_success_step", "steps_after_first_success",
    "termination_reason", "retry", "retry_trigger", "max_retries_per_subtask",
    "retry_attempts", "retry_counts_by_subtask", "planner_subgoals_visited",
    "post_success_reactivation", "injection_count", "subtasks", "plan",
)

def project(row, keys):
    return {key: row.get(key) for key in keys}

base_decisions = rows(baseline / "decisions.jsonl")
residual_decisions = rows(residual / "decisions.jsonl")
base_episodes = rows(baseline / "episodes.jsonl")
residual_episodes = rows(residual / "episodes.jsonl")
decision_equal = [project(row, decision_keys) for row in base_decisions] == [
    project(row, decision_keys) for row in residual_decisions
]
episode_equal = [project(row, episode_keys) for row in base_episodes] == [
    project(row, episode_keys) for row in residual_episodes
]
payload = {
    "schema_version": 1,
    "base_seed": 30007,
    "case": "Ideal/case1",
    "forced_abstention_margin": 1000000,
    "decision_rows": len(base_decisions),
    "episode_rows": len(base_episodes),
    "decision_projection_equal": decision_equal,
    "episode_projection_equal": episode_equal,
    "complete": decision_equal and episode_equal,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
if not payload["complete"]:
    raise SystemExit("residual abstention diverged from B-retry")
print(json.dumps(payload, indent=2))
PY

stop_server
