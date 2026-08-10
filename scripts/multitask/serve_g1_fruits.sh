#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"
GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
OUTPUT_DIR="${GR00T_MULTITASK_OUTPUT:-${PROJECT_ROOT}/checkpoints/groot-g1-fruits-multitask}"
POLICY_PORT="${POLICY_PORT:-5550}"
MODE="${GR00T_MULTITASK_MODE:-base}"

case "${MODE}" in
    base)
        checkpoint="${GR00T_MULTITASK_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/nvidia-gr00t-n1.7-3b}"
        embodiment_tag="REAL_G1"
        ;;
    finetuned)
        checkpoint="${GR00T_MULTITASK_CHECKPOINT:-}"
        if [[ -z "${checkpoint}" ]]; then
            checkpoint="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)"
        fi
        embodiment_tag="NEW_EMBODIMENT"
        ;;
    *)
        echo "GR00T_MULTITASK_MODE must be 'base' or 'finetuned'" >&2
        exit 2
        ;;
esac
[[ -n "${checkpoint}" && -f "${checkpoint}/config.json" ]] || {
    echo "Checkpoint is missing for mode=${MODE}: ${checkpoint:-<none>}" >&2
    exit 1
}
"${GR00T_DIR}/.venv/bin/python" "${PROJECT_ROOT}/scripts/groot/check_model_access.py" \
    --repo nvidia/Cosmos-Reason2-2B

echo "Serving GR00T mode=${MODE}, tag=${embodiment_tag}: ${checkpoint}"
cd "${GR00T_DIR}"
exec .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path "${checkpoint}" \
    --embodiment-tag "${embodiment_tag}" \
    --use-sim-policy-wrapper \
    --device cuda:0 \
    --port "${POLICY_PORT}"
