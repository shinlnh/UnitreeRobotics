#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
mkdir -p "$(dirname "${SONIC_DIR}")"

if [[ ! -d "${SONIC_DIR}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none https://github.com/NVlabs/GR00T-WholeBodyControl.git "${SONIC_DIR}"
fi
git -C "${SONIC_DIR}" fetch --depth 1 origin "${SONIC_REF}"
git -C "${SONIC_DIR}" checkout --detach "${SONIC_REF}"

if [[ "${SONIC_PULL_LFS:-1}" == "1" ]]; then
    command -v git-lfs >/dev/null || {
        echo "git-lfs is required for SONIC robot meshes and released controller files." >&2
        exit 1
    }
    git -C "${SONIC_DIR}" lfs pull --exclude="motionbricks/out/**"
fi

cat <<EOF
GEAR-SONIC source is pinned at ${SONIC_REF}.
The released controller, MuJoCo simulator, data collection, and inference tools
use separate environments. SONIC RL training requires Python 3.11 and Isaac Lab
2.3+, so it is intentionally not installed into this Python 3.12 environment.
EOF
