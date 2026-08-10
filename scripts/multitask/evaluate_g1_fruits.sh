#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
DATASET_ROOT="${GR00T_MULTITASK_DATASET_ROOT:-${PROJECT_ROOT}/datasets/g1_fruits_multitask_hf}"
POLICY_PORT="${POLICY_PORT:-5550}"
EVAL_TRAJECTORIES="${EVAL_TRAJECTORIES:-0 1 2 3}"

for name in apple pear grapes starfruit; do
    echo "Open-loop evaluation: ${name}"
    # shellcheck disable=SC2086
    "${GR00T_DIR}/.venv/bin/python" "${GR00T_DIR}/gr00t/eval/open_loop_eval.py" \
        --host 127.0.0.1 --port "${POLICY_PORT}" \
        --dataset-path "${DATASET_ROOT}/g1-pick-${name}" \
        --embodiment-tag NEW_EMBODIMENT \
        --execution-horizon 8 \
        --steps 160 \
        --traj-ids ${EVAL_TRAJECTORIES}
done
