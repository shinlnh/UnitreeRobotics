#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${PROJECT_ROOT}/.deps/IsaacLab/.venv/bin/python"
N15_PYTHON="${PROJECT_ROOT}/.deps/Isaac-GR00T-N1.5/.venv/bin/python"
POLICY_LOG="${PROJECT_ROOT}/outputs/locomanip_policy.log"
DEMO_LOG="${PROJECT_ROOT}/outputs/commanded_fetch_demo.log"
PORT="${LOCOMANIP_POLICY_PORT:-5556}"
INSTRUCTION="${INSTRUCTION:-Hãy lấy vô lăng trên bàn đầu và mang tới bàn giao.}"
POLICY_PID=""

test -x "${ISAAC_PYTHON}" || { echo "Run 'make setup-isaaclab' first" >&2; exit 1; }
test -x "${N15_PYTHON}" || { echo "Run 'make setup-locomanip' first" >&2; exit 1; }
mkdir -p "${PROJECT_ROOT}/outputs"

cleanup() {
    if [[ -n "${POLICY_PID}" ]] && kill -0 "${POLICY_PID}" 2>/dev/null; then
        kill "${POLICY_PID}" 2>/dev/null || true
        wait "${POLICY_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if ! "${ISAAC_PYTHON}" "${PROJECT_ROOT}/scripts/locomanip/ping_policy.py" --port "${PORT}"; then
    "${N15_PYTHON}" "${PROJECT_ROOT}/scripts/locomanip/serve_policy.py" --port "${PORT}" \
        >"${POLICY_LOG}" 2>&1 &
    POLICY_PID="$!"
    echo "Loading GR00T locomanipulation policy (log: ${POLICY_LOG}) ..."
    for _ in $(seq 1 300); do
        if "${ISAAC_PYTHON}" "${PROJECT_ROOT}/scripts/locomanip/ping_policy.py" --port "${PORT}"; then
            break
        fi
        if ! kill -0 "${POLICY_PID}" 2>/dev/null; then
            tail -80 "${POLICY_LOG}" >&2
            exit 1
        fi
        sleep 1
    done
fi

set +e
"${ISAAC_PYTHON}" "${PROJECT_ROOT}/scripts/locomanip/run_commanded_fetch.py" \
    --policy-port "${PORT}" --instruction "${INSTRUCTION}" "$@" 2>&1 | tee "${DEMO_LOG}"
RUN_STATUS="${PIPESTATUS[0]}"
set -e

# Kit can terminate the Python process from simulation_app.close() before
# SystemExit propagates.  Keep the one-command demo honest by treating the
# runner's explicit terminal marker as authoritative.
if grep -q '^\[FAIL\]' "${DEMO_LOG}"; then
    exit 1
fi
exit "${RUN_STATUS}"
