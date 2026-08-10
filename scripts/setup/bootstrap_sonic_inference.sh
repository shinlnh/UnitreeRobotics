#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"

UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
fi
[[ -n "${UV_BIN}" ]] || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -d "${SONIC_DIR}/gear_sonic" ]] || { echo "Run: make setup-sonic" >&2; exit 1; }
[[ -d "${GR00T_DIR}/gr00t" ]] || { echo "Run: make setup-groot" >&2; exit 1; }

"${UV_BIN}" python install 3.10
"${UV_BIN}" venv --clear "${SONIC_DIR}/.venv_inference" --python 3.10 \
    --prompt gear_sonic_inference

# Upstream's inference extra currently declares distribution name Isaac-GR00T,
# while the pinned repository publishes package metadata as gr00t. New uv
# versions reject that mismatch. Install only robot-side client dependencies;
# the project supplies a wire-compatible numeric-only PolicyClient, so no
# training/CUDA stack is needed in this Python 3.10 environment.
"${UV_BIN}" pip install --python "${SONIC_DIR}/.venv_inference/bin/python" \
    --no-deps -e "${SONIC_DIR}/gear_sonic"
"${UV_BIN}" pip install --python "${SONIC_DIR}/.venv_inference/bin/python" \
    'numpy==1.26.4' 'scipy==1.15.3' joblib tqdm easydict loguru \
    pyzmq msgpack msgpack-numpy pin tyro opencv-python

"${SONIC_DIR}/.venv_inference/bin/python" - <<'PY'
from gear_sonic.utils.inference import vla_utils  # noqa: F401
import msgpack_numpy  # noqa: F401
import zmq  # noqa: F401

print("SONIC inference client dependencies are ready.")
PY
