#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKERS="${A1_CONVERSION_WORKERS:-10}"

PYTHONPATH="${PROJECT_ROOT}/src" "${PROJECT_ROOT}/.venv-a0/bin/python" \
  -m unitree_gr00t.a1_data \
  --robocerebra-source "${PROJECT_ROOT}/.upstream/RoboCerebra" \
  --benchmark-dir "${PROJECT_ROOT}/.cache/robocerebra/bench" \
  --source "${PROJECT_ROOT}/.cache/robocerebra/train/RoboCerebra_trainset" \
  --manifest "${PROJECT_ROOT}/.cache/robocerebra/train/RoboCerebra_trainset/trainingset.json" \
  --expected-manifest-sha256 "3211be95643a2ff8fdda2253267137272617821ac93ae827fd12355e7a39d46f" \
  --destination "${PROJECT_ROOT}/outputs/robocerebra/gr00t-lerobot-v2" \
  --source-revision "5d2e1e361bf65aabbe4d18179515f5a10936cc96" \
  --expected-manifest-rows 1000 \
  --expected-episodes 995 \
  --workers "${WORKERS}" \
  --fps 20 \
  --resume

echo "A1 GR00T LeRobot dataset is ready"
