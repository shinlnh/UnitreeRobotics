#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-.venv-a0/bin/python}"
WORKERS="${WORKERS:-8}"
SHARDS_PER_WORKER="${SHARDS_PER_WORKER:-2}"
OFFSET="${OFFSET:-32}"
SELECTION_SEED="${SELECTION_SEED:-89007}"
BASE_SEED="${BASE_SEED:-89007}"
PORT="${PORT:-5559}"
FRONTIERS="${FRONTIERS:-outputs/robocerebra/resolve-expert-frontiers-v2}"
RUNTIME_MANIFEST="${RUNTIME_MANIFEST:-outputs/robocerebra/ctr-residual-live-corpus-v1/train-seed12007-B-retry-H16/manifest.json}"
ROBOCEREBRA="${ROBOCEREBRA:-.upstream/RoboCerebra}"
DESTINATION="${DESTINATION:-outputs/robocerebra/resolve-frontier-full-offset${OFFSET}}"

if (( WORKERS < 1 || SHARDS_PER_WORKER < 1 )); then
  echo "WORKERS and SHARDS_PER_WORKER must be positive" >&2
  exit 2
fi

TOTAL="$({ PYTHONPATH=src "${PYTHON}" - "${FRONTIERS}" "${OFFSET}" "${SELECTION_SEED}" <<'PY'
import sys
from pathlib import Path

from unitree_gr00t.resolve_frontier_rollout import _read_jsonl, flatten_frontier_anchors

root = Path(sys.argv[1])
rows = _read_jsonl(root / "frontiers.jsonl")
selected = flatten_frontier_anchors(
    rows,
    split="train",
    anchor_offset=int(sys.argv[2]),
    selection_seed=int(sys.argv[3]),
    start=0,
    limit=None,
)
print(len(selected))
PY
} | tail -1)"

SHARD_COUNT=$(( WORKERS * SHARDS_PER_WORKER ))
PER_SHARD=$(( (TOTAL + SHARD_COUNT - 1) / SHARD_COUNT ))
shards=()
status=0
for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
  start=$(( shard_index * PER_SHARD ))
  if (( start >= TOTAL )); then
    break
  fi
  count=$(( TOTAL - start ))
  if (( count > PER_SHARD )); then
    count="${PER_SHARD}"
  fi
  shard="${DESTINATION}-shard${shard_index}"
  shards+=("${shard}")

  # Keep a dynamic worker pool instead of assigning one large static shard to
  # each worker.  Short workers immediately take another shard, which avoids
  # the long-tail CPU/GPU starvation caused by variable episode lengths.
  while (( $(jobs -pr | wc -l) >= WORKERS )); do
    if ! wait -n; then
      status=1
    fi
  done
  RESOLVE_LIBERO_CONFIG=outputs/robocerebra/resolve-libero-config \
  PYTHONPATH=src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "${PYTHON}" -m unitree_gr00t.resolve_frontier_rollout \
      --frontiers "${FRONTIERS}" \
      --runtime-manifest "${RUNTIME_MANIFEST}" \
      --robocerebra-source "${ROBOCEREBRA}" \
      --output "${shard}" \
      --split train \
      --anchor-offset "${OFFSET}" \
      --selection-seed "${SELECTION_SEED}" \
      --anchor-start "${start}" \
      --limit-anchors "${count}" \
      --policy-port "${PORT}" \
      --base-seed "${BASE_SEED}" \
      >"/tmp/resolve-frontier-shard${shard_index}.log" 2>&1 &
done

while (( $(jobs -pr | wc -l) > 0 )); do
  if ! wait -n; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "at least one RESOLVE frontier shard failed" >&2
  exit 2
fi

PYTHONPATH=src "${PYTHON}" -m unitree_gr00t.resolve_frontier_merge \
  --shards "${shards[@]}" \
  --output "${DESTINATION}"
