#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

GR00T_DIR="${GR00T_ROOT:-${PROJECT_ROOT}/.deps/Isaac-GR00T}"
mkdir -p "$(dirname "${GR00T_DIR}")"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

if [[ ! -d "${GR00T_DIR}/.git" ]]; then
    git clone --filter=blob:none https://github.com/NVIDIA/Isaac-GR00T.git "${GR00T_DIR}"
fi

git -C "${GR00T_DIR}" fetch --depth 1 origin "${GR00T_REF}"
git -C "${GR00T_DIR}" checkout --detach "${GR00T_REF}"
uv sync --project "${GR00T_DIR}"

# TorchCodec 0.9 is the PyTorch 2.9-compatible release with FFmpeg 8 support.
# Ubuntu's FFmpeg 6 split packages can omit libavdevice, which TorchCodec links
# even for file decoding. Keep that small runtime library project-local.
uv pip install --python "${GR00T_DIR}/.venv/bin/python" "torchcodec==${TORCHCODEC_VERSION}"
FFMPEG_RUNTIME="${PROJECT_ROOT}/.deps/ffmpeg6-runtime"
LOCAL_AVDEVICE="${FFMPEG_RUNTIME}/usr/lib/x86_64-linux-gnu/libavdevice.so.60"
if [[ ! -e "${LOCAL_AVDEVICE}" ]] && ! ldconfig -p 2>/dev/null | grep -q 'libavdevice.so.60'; then
    command -v apt >/dev/null || { echo "apt is required to download libavdevice60" >&2; exit 1; }
    command -v dpkg-deb >/dev/null || { echo "dpkg-deb is required" >&2; exit 1; }
    mkdir -p "${FFMPEG_RUNTIME}"
    DOWNLOAD_DIR="$(mktemp -d)"
    trap 'rm -rf "${DOWNLOAD_DIR}"' EXIT
    (
        cd "${DOWNLOAD_DIR}"
        apt download libavdevice60
        dpkg-deb -x libavdevice60_*.deb "${FFMPEG_RUNTIME}"
    )
    rm -rf "${DOWNLOAD_DIR}"
    trap - EXIT
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/groot/groot_runtime_env.sh"
"${GR00T_DIR}/.venv/bin/python" -c "import torchcodec; assert torchcodec.__version__ == '${TORCHCODEC_VERSION}'"

# LeApp is NVIDIA's runtime for the exported ApplePnP ONNX graph. Keep it at a
# commit pin and use the final CUDA-12 ONNX Runtime release supported here.
LEAPP_DIR="${PROJECT_ROOT}/.deps/leapp"
if [[ ! -d "${LEAPP_DIR}/.git" ]]; then
    git clone https://github.com/nvidia-isaac/leapp.git "${LEAPP_DIR}"
fi
git -C "${LEAPP_DIR}" fetch --depth 1 origin "${LEAPP_REF}"
git -C "${LEAPP_DIR}" checkout --detach "${LEAPP_REF}"
uv pip install --python "${GR00T_DIR}/.venv/bin/python" --no-deps -e "${LEAPP_DIR}"
uv pip install --python "${GR00T_DIR}/.venv/bin/python" \
    "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"
"${GR00T_DIR}/.venv/bin/python" - <<PY
import leapp
import onnxruntime as ort
assert leapp.__version__ == "${LEAPP_VERSION}"
assert ort.__version__ == "${ONNXRUNTIME_GPU_VERSION}"
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY

echo "GR00T ${GR00T_RELEASE} @ ${GR00T_REF} is ready in its own Torch 2.9 environment."
