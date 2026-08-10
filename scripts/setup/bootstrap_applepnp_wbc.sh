#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
PYTHON="${SONIC_DIR}/.venv_sim/bin/python"
ROBOSUITE_DIR="${PROJECT_ROOT}/.deps/robosuite-g1"

[[ -x "${PYTHON}" ]] || {
    echo "SONIC simulation environment is missing. Run: make setup-sonic-sim" >&2
    exit 1
}

if [[ ! -d "${ROBOSUITE_DIR}/.git" ]]; then
    git clone --filter=blob:none --branch leo/support_g1_locomanip \
        https://github.com/xieleo5/robosuite.git "${ROBOSUITE_DIR}"
fi
git -C "${ROBOSUITE_DIR}" fetch --depth 1 origin "${ROBOSUITE_G1_REF}"
git -C "${ROBOSUITE_DIR}" checkout --detach "${ROBOSUITE_G1_REF}"

# The upstream whole-body stack's Dockerfile requires this G1-enabled
# robosuite fork. RoboCasa itself pins MuJoCo 3.2.6, so install the fork with
# no dependency resolution after installing the exact runtime set.
uv pip install --python "${PYTHON}" \
    -e "${SONIC_DIR}/decoupled_wbc/dexmg/gr00trobocasa" \
    "gymnasium==1.2.3" "onnxruntime==${ONNXRUNTIME_GPU_VERSION}" \
    "mink==0.0.5" matplotlib meshcat meshcat-shapes \
    'qpsolvers[osqp,quadprog]' pin-pink "rerun-sdk==0.21.0"
uv pip install --python "${PYTHON}" --no-deps -e "${ROBOSUITE_DIR}"

(
    cd "${SONIC_DIR}"
    MUJOCO_GL=egl .venv_sim/bin/python - <<'PY'
import gymnasium as gym
import decoupled_wbc.control.envs.robocasa.sync_env  # noqa: F401

task = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
assert task in gym.registry, task
print(f"ApplePnP GEAR-WBC simulator registered: {task}")
PY
)
