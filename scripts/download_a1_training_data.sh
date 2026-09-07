#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="qiukingballball/RoboCerebra"
REVISION="5d2e1e361bf65aabbe4d18179515f5a10936cc96"
DESTINATION="${PROJECT_ROOT}/.cache/robocerebra/train"
WORKERS="${A1_DOWNLOAD_WORKERS:-32}"

command -v hf >/dev/null || {
  echo "hf CLI is required" >&2
  exit 1
}

# A1 re-renders both policy cameras from simulator states, so the 137 GB RLDS
# exports and source MP4/PNG previews are deliberately not downloaded.
hf download "${DATASET}" \
  --type dataset \
  --revision "${REVISION}" \
  --include "RoboCerebra_trainset/trainingset.json" \
  --include "RoboCerebra_trainset/**/*.hdf5" \
  --include "RoboCerebra_trainset/**/*.bddl" \
  --local-dir "${DESTINATION}" \
  --max-workers "${WORKERS}" \
  --format agent

echo "A1 raw training states ready at ${DESTINATION}"
