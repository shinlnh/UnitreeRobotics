#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_VENV="${PROJECT_ROOT}/.venv-a0"
ROBOCEREBRA_LIBERO="${PROJECT_ROOT}/.upstream/RoboCerebra/LIBERO"

command -v uv >/dev/null || {
  echo "uv is required" >&2
  exit 1
}

if [[ ! -d "${ROBOCEREBRA_LIBERO}" ]]; then
  echo "RoboCerebra is missing; run scripts/bootstrap_upstreams.sh first." >&2
  exit 1
fi

if [[ ! -x "${EVAL_VENV}/bin/python" ]]; then
  uv venv "${EVAL_VENV}" --python 3.10 --no-project
fi
uv pip install --python "${EVAL_VENV}/bin/python" \
  "numpy==1.26.4" \
  "robosuite==1.4.0" \
  "mujoco>=2.3.7,<3" \
  "bddl==1.0.1" \
  "gym==0.25.2" \
  "h5py>=3.10,<4" \
  "pandas>=2.2,<3" \
  "pyarrow>=18,<24" \
  "future>=0.18,<2" \
  "cloudpickle>=2.1,<4" \
  "easydict>=1.9,<2" \
  "einops>=0.4,<1" \
  "opencv-python==4.6.0.66" \
  "matplotlib>=3.5,<4" \
  "imageio>=2.31,<3" \
  "pyyaml>=6,<7" \
  "termcolor>=2,<4" \
  "pyzmq>=25,<28" \
  "msgpack>=1,<2"
uv pip install --python "${EVAL_VENV}/bin/python" \
  --editable "${ROBOCEREBRA_LIBERO}" \
  --no-deps

PYTHONPATH="${PROJECT_ROOT}/src" "${EVAL_VENV}/bin/python" -c \
  "import msgpack, mujoco, numpy, robosuite, zmq; import unitree_gr00t.a0_eval; print('A0 evaluator environment is ready')"
