#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/sonic/sonic_deploy_env.sh"

SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
DEPLOY_DIR="${SONIC_DIR}/gear_sonic_deploy"
BINARY="${DEPLOY_DIR}/target/release/g1_deploy_onnx_ref"

[[ -x "${BINARY}" ]] || {
    echo "SONIC deploy binary is missing. Run: make setup-sonic-deploy" >&2
    exit 1
}

cd "${DEPLOY_DIR}"
exec "${BINARY}" \
    "${SONIC_SIM_INTERFACE:-lo}" \
    policy/sonic_v1_1/model_decoder.onnx \
    reference/example/ \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --encoder-file policy/sonic_v1_1/model_encoder.onnx \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager \
    --output-type all \
    --zmq-host localhost \
    --disable-crc-check \
    "$@"
