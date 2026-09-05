#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${PROJECT_ROOT}/.upstream"
INSTALL=false

if [[ "${1:-}" == "--install" ]]; then
  INSTALL=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--install]" >&2
  exit 2
fi

ISAAC_URL="https://github.com/NVIDIA/Isaac-GR00T.git"
ISAAC_REV="51d4c89f72fda44cbf77285c6a8114b52676b8a1"
SONIC_URL="https://github.com/NVlabs/GR00T-WholeBodyControl.git"
SONIC_REV="a0732b642c0333077e127a2f56ab0014c196bca4"

clone_pinned() {
  local url="$1"
  local destination="$2"
  local revision="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --recurse-submodules "${url}" "${destination}"
  fi
  git -C "${destination}" fetch origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
  git -C "${destination}" submodule update --init --recursive
}

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
mkdir -p "${UPSTREAM_ROOT}"
clone_pinned "${ISAAC_URL}" "${UPSTREAM_ROOT}/Isaac-GR00T" "${ISAAC_REV}"
clone_pinned "${SONIC_URL}" "${UPSTREAM_ROOT}/GR00T-WholeBodyControl" "${SONIC_REV}"

if [[ "${INSTALL}" == true ]]; then
  command -v uv >/dev/null || { echo "uv is required for installation" >&2; exit 1; }
  (cd "${UPSTREAM_ROOT}/Isaac-GR00T" && uv sync --python 3.12)
  (cd "${UPSTREAM_ROOT}/GR00T-WholeBodyControl" && bash install_scripts/install_mujoco_sim.sh)
  (cd "${UPSTREAM_ROOT}/GR00T-WholeBodyControl" && bash install_scripts/install_inference.sh)
  echo "Build the C++ controller next: cd ${UPSTREAM_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy && just build"
fi

echo "Pinned upstream checkouts are ready in ${UPSTREAM_ROOT}"
