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
ISAAC_REV="4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e"
SONIC_URL="https://github.com/NVlabs/GR00T-WholeBodyControl.git"
SONIC_REV="a0732b642c0333077e127a2f56ab0014c196bca4"
ROBOCEREBRA_URL="https://github.com/buaa-colalab/RoboCerebra.git"
ROBOCEREBRA_REV="2573426c13dfcd5e7d7831c15587b058aaa1c0c0"
SPARKVLA_URL="https://github.com/huhuhushou/SparkVLA.git"
SPARKVLA_REV="ae90ec94ccc77a1e6745e4f5bcc94e05502d8cbf"

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
clone_pinned "${ISAAC_URL}" "${UPSTREAM_ROOT}/Isaac-GR00T-N1.7" "${ISAAC_REV}"
clone_pinned "${SONIC_URL}" "${UPSTREAM_ROOT}/GR00T-WholeBodyControl" "${SONIC_REV}"
clone_pinned "${ROBOCEREBRA_URL}" "${UPSTREAM_ROOT}/RoboCerebra" "${ROBOCEREBRA_REV}"
clone_pinned "${SPARKVLA_URL}" "${UPSTREAM_ROOT}/SparkVLA" "${SPARKVLA_REV}"

if [[ "${INSTALL}" == true ]]; then
  command -v uv >/dev/null || { echo "uv is required for installation" >&2; exit 1; }
  (cd "${UPSTREAM_ROOT}/Isaac-GR00T-N1.7" && uv sync --python 3.12)
  (cd "${UPSTREAM_ROOT}/GR00T-WholeBodyControl" && bash install_scripts/install_mujoco_sim.sh)
  (cd "${UPSTREAM_ROOT}/GR00T-WholeBodyControl" && bash install_scripts/install_inference.sh)
  echo "Build the C++ controller next: cd ${UPSTREAM_ROOT}/GR00T-WholeBodyControl/gear_sonic_deploy && just build"
fi

echo "Pinned upstream checkouts are ready in ${UPSTREAM_ROOT}"
