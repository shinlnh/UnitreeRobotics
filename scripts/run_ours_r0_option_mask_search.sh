#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
mkdir -p artifacts/Ours/search/R0-residual-v9-option-mask

PYTHONPATH=src .upstream/Isaac-GR00T-N1.7/.venv/bin/python \
  -m unitree_gr00t.ours_mask_search \
  --checkpoint-root checkpoints/robocerebra/GR00T-RC-CTR-search/R0-residual-v9-override-weight \
  --output artifacts/Ours/search/R0-residual-v9-option-mask/registry.json \
  --expected-checkpoints 24 \
  --batch-size 256 \
  --device cuda:0 \
  >artifacts/Ours/search/R0-residual-v9-option-mask/search.log
