#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

ISAACLAB_DIR="${ISAACLAB_ROOT:-${PROJECT_ROOT}/.deps/IsaacLab}"
ISAACLAB_ENV="${ISAACLAB_DIR}/.venv"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

mkdir -p "$(dirname "${ISAACLAB_DIR}")"
if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
    git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

git -C "${ISAACLAB_DIR}" fetch --depth 1 origin "${ISAAC_LAB_REF}"
git -C "${ISAACLAB_DIR}" checkout --detach "${ISAAC_LAB_REF}"

if [[ ! -x "${ISAACLAB_ENV}/bin/python" ]]; then
    uv venv --python 3.12 --seed "${ISAACLAB_ENV}"
fi

# The upstream checkout carries the lock for this exact commit. Keep it immutable so a
# newer uv release cannot rewrite marker normalization inside the vendored repository.
uv sync --frozen --project "${ISAACLAB_DIR}" --python "${ISAACLAB_ENV}/bin/python" --extra isaacsim --extra rsl-rl
uv pip install --python "${ISAACLAB_ENV}/bin/python" -e "${PROJECT_ROOT}/source/unitree_rl_groot[groot-client,dev]"

echo "Isaac Lab ${ISAAC_LAB_VERSION} @ ${ISAAC_LAB_REF} is ready."
echo "Run: ${ISAACLAB_ENV}/bin/python ${PROJECT_ROOT}/scripts/doctor.py --strict"
