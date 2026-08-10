#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

GR00T_DIR="${GR00T_LOCOMANIP_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T-N1.5}"
ISAAC_PYTHON="${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python"
SDPA_PATCH="${PROJECT_ROOT}/.deps/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/gr00t/no_flash_attn.patch"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
test -x "${ISAAC_PYTHON}" || { echo "Run 'make setup-isaaclab' first" >&2; exit 1; }

if [[ ! -d "${GR00T_DIR}/.git" ]]; then
    git clone --filter=blob:none https://github.com/NVIDIA/Isaac-GR00T.git "${GR00T_DIR}"
fi
git -C "${GR00T_DIR}" fetch --depth 1 origin "${GR00T_LOCOMANIP_REF}"
git -C "${GR00T_DIR}" checkout --detach "${GR00T_LOCOMANIP_REF}"

# The official Isaac Lab compatibility patch selects PyTorch SDPA.  This is
# required on Blackwell because the N1.5 flash-attn pin has no matching wheel.
if git -C "${GR00T_DIR}" apply --check "${SDPA_PATCH}" 2>/dev/null; then
    git -C "${GR00T_DIR}" apply "${SDPA_PATCH}"
elif ! git -C "${GR00T_DIR}" apply --reverse --check "${SDPA_PATCH}" 2>/dev/null; then
    echo "N1.5 SDPA patch does not match pinned checkout" >&2
    exit 1
fi

if [[ ! -x "${GR00T_DIR}/.venv/bin/python" ]]; then
    PYTHON_310="$(uv python find 3.10)"
    uv venv --python "${PYTHON_310}" "${GR00T_DIR}/.venv"
fi
uv pip install --python "${GR00T_DIR}/.venv/bin/python" -e "${GR00T_DIR}[base]" msgpack pyzmq
# N1.5's historical torch 2.5 wheel predates Blackwell (sm_120).  Keep the
# N1.5 model code but use NVIDIA GR00T N1.7's verified CUDA 12.8 Torch pair.
uv pip install --python "${GR00T_DIR}/.venv/bin/python" \
    --index https://download.pytorch.org/whl/cu128 \
    "torch==${GR00T_LOCOMANIP_TORCH_VERSION}" \
    "torchvision==${GR00T_LOCOMANIP_TORCHVISION_VERSION}"
uv pip install --python "${ISAAC_PYTHON}" -e "${PROJECT_ROOT}/.deps/IsaacLab/source/isaaclab_mimic"
uv pip install --python "${ISAAC_PYTHON}" -e "${PROJECT_ROOT}/.deps/IsaacLab/source/isaaclab_teleop"
uv pip install --python "${ISAAC_PYTHON}" -e "${PROJECT_ROOT}/source/unitree_rl_groot[groot-client]"

"${GR00T_DIR}/.venv/bin/python" - <<'PY'
import torch
from gr00t.model.policy import Gr00tPolicy

assert torch.cuda.is_available()
print(f"GR00T N1.5 runtime ready: torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
PY

echo "Pinned GR00T ${GR00T_LOCOMANIP_RELEASE} runtime is ready at ${GR00T_DIR}."
