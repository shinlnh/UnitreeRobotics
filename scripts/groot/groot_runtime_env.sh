#!/usr/bin/env bash
# Runtime libraries kept inside the project for GR00T/TorchCodec.

GROOT_RUNTIME_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GROOT_FFMPEG_RUNTIME="${GROOT_FFMPEG_RUNTIME:-${GROOT_RUNTIME_PROJECT_ROOT}/.deps/ffmpeg6-runtime/usr/lib/x86_64-linux-gnu}"
if [[ -d "${GROOT_FFMPEG_RUNTIME}" ]]; then
    export LD_LIBRARY_PATH="${GROOT_FFMPEG_RUNTIME}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# PyTorch wheels keep CUDA 12 libraries (notably cuDNN) inside site-packages.
# ONNX Runtime loads them through the dynamic linker instead of through torch,
# so expose those project-local directories without requiring a system install.
for GROOT_NVIDIA_LIB in \
    "${GROOT_RUNTIME_PROJECT_ROOT}"/.deps/Isaac-GR00T/.venv/lib/python*/site-packages/nvidia/*/lib; do
    if [[ -d "${GROOT_NVIDIA_LIB}" ]]; then
        export LD_LIBRARY_PATH="${GROOT_NVIDIA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
done
unset GROOT_RUNTIME_PROJECT_ROOT GROOT_FFMPEG_RUNTIME GROOT_NVIDIA_LIB
