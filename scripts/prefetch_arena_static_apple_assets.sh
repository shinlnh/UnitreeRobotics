#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA_DIR="$PROJECT_ROOT/.upstream/IsaacLab-Arena"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/shin/IsaacLab}"
ARENA_ASSET_CACHE_DIR="${ARENA_ASSET_CACHE_DIR:-$PROJECT_ROOT/.cache/isaaclab-assets}"
ARENA_ASSET_DOWNLOAD_WORKERS="${ARENA_ASSET_DOWNLOAD_WORKERS:-16}"

arena_lab_paths="$(find "$ARENA_DIR/submodules/IsaacLab/source" \
    -mindepth 1 -maxdepth 1 -type d -name 'isaaclab*' -printf '%p:')"

export VIRTUAL_ENV="$ISAACLAB_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export ISAAC_PATH="$ISAACLAB_DIR/_isaac_sim"
export CARB_APP_PATH="$ISAAC_PATH/kit"
export EXP_PATH="$ISAAC_PATH/apps"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${arena_lab_paths}${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ISAACLAB_ASSET_CACHE_DIR="$ARENA_ASSET_CACHE_DIR"

mkdir -p "$ARENA_ASSET_CACHE_DIR"

# Isaac Sim's environment script requires bash because it uses BASH_SOURCE.
# shellcheck disable=SC1091
source "$ISAAC_PATH/setup_python_env.sh"

exec "$VIRTUAL_ENV/bin/python" "$PROJECT_ROOT/scripts/prefetch_arena_static_apple_assets.py" \
    --cache-dir "$ARENA_ASSET_CACHE_DIR" \
    --workers "$ARENA_ASSET_DOWNLOAD_WORKERS" \
    "$@"
