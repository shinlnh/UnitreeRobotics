#!/usr/bin/env bash

# Runtime paths for the project-local SONIC C++ toolchain. Safe to source before
# the toolchain exists; the preflight check will then report what is missing.
SONIC_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SONIC_DEPLOY_TOOLS="${SONIC_DEPLOY_TOOLS:-${SONIC_PROJECT_ROOT}/.deps/sonic-deploy}"

export PATH="${SONIC_DEPLOY_TOOLS}/bin:${PATH}"
export ONNXRUNTIME_ROOT="${SONIC_DEPLOY_TOOLS}/onnxruntime"
export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib:${SONIC_DEPLOY_TOOLS}/sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ESPEAK_DATA_PATH="${SONIC_DEPLOY_TOOLS}/sysroot/usr/lib/x86_64-linux-gnu/espeak-data"

unset SONIC_PROJECT_ROOT
