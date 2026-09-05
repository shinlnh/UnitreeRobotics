#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_NAME="arena-static-apple-demo"

if systemctl --user is-active --quiet "$UNIT_NAME.service"; then
    echo "The one-shot Isaac Arena demo is already running."
    echo "Stop it with: systemctl --user stop $UNIT_NAME.service"
    exit 0
fi

exec systemd-run --user \
    --unit="$UNIT_NAME" \
    --collect \
    --property=MemoryHigh=22G \
    --property=MemoryMax=24G \
    --property=OOMScoreAdjust=700 \
    --working-directory="$PROJECT_ROOT" \
    /usr/bin/env ARENA_RUN_NAME=static_apple_demo_once \
    "$PROJECT_ROOT/scripts/run_arena_static_apple.sh" demo-once
