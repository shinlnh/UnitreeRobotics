#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONIC_ROOT="${PROJECT_ROOT}/.upstream/GR00T-WholeBodyControl"

command -v hf >/dev/null || { echo "Install the Hugging Face CLI first." >&2; exit 1; }
hf auth whoami >/dev/null
hf download nvidia/Cosmos-Reason2-2B --include config.json --dry-run --format json >/dev/null || {
  echo "Request access to nvidia/Cosmos-Reason2-2B and authenticate with: hf auth login" >&2
  exit 1
}

if [[ ! -f "${SONIC_ROOT}/download_from_hf.py" ]]; then
  echo "Run scripts/bootstrap_upstreams.sh first." >&2
  exit 1
fi

cd "${SONIC_ROOT}"
python3 download_from_hf.py --sonic-v1-1
hf download nvidia/GR00T-N1.7-3B --dry-run

echo "SONIC v1.1 is installed. GR00T base-model download was validated with --dry-run."
