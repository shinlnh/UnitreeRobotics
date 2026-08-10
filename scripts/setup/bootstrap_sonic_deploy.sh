#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/configs/setup/versions.env"

SONIC_DIR="${SONIC_ROOT:-${PROJECT_ROOT}/.deps/GR00T-WholeBodyControl}"
DEPLOY_DIR="${SONIC_DIR}/gear_sonic_deploy"
TOOLS_DIR="${SONIC_DEPLOY_TOOLS:-${PROJECT_ROOT}/.deps/sonic-deploy}"
DOWNLOAD_DIR="${TOOLS_DIR}/downloads"
SYSROOT="${TOOLS_DIR}/sysroot"
BIN_DIR="${TOOLS_DIR}/bin"
ONNX_DIR="${TOOLS_DIR}/onnxruntime"

[[ -f "${DEPLOY_DIR}/CMakeLists.txt" ]] || {
    echo "SONIC source is missing. Run: make setup-sonic" >&2
    exit 1
}
[[ "$(dpkg --print-architecture)" == "amd64" ]] || {
    echo "The project-local deploy bootstrap currently supports Ubuntu amd64 only." >&2
    echo "On Jetson/arm64, use the official gear_sonic_deploy/scripts/install_deps.sh." >&2
    exit 1
}

mkdir -p "${DOWNLOAD_DIR}" "${SYSROOT}" "${BIN_DIR}"

extract_deb() {
    local package="$1"
    local package_spec="${2:-${package}}"
    local deb
    deb="$(find "${DOWNLOAD_DIR}" -maxdepth 1 -type f -name "${package}_*.deb" -print -quit)"
    if [[ -z "${deb}" ]]; then
        echo "Downloading ${package}..."
        (cd "${DOWNLOAD_DIR}" && apt-get download "${package_spec}")
        deb="$(find "${DOWNLOAD_DIR}" -maxdepth 1 -type f -name "${package}_*.deb" -print -quit)"
    fi
    [[ -n "${deb}" ]] || {
        echo "Could not download ${package}. Check apt repositories/network." >&2
        exit 1
    }
    dpkg-deb -x "${deb}" "${SYSROOT}"
}

# Header-only packages plus the unversioned libzmq linker symlink and a local
# tmux binary. Runtime dependencies for tmux/libzmq are standard Ubuntu libs.
for package in tmux libutempter0 espeak espeak-data libespeak1 libportaudio2 libmsgpack-dev libmsgpack-cxx-dev libzmq5 libzmq3-dev cppzmq-dev nlohmann-json3-dev; do
    extract_deb "${package}"
done
extract_deb libnvinfer-headers-dev "libnvinfer-headers-dev=${SONIC_TENSORRT_VERSION}"
extract_deb libnvonnxparsers-dev "libnvonnxparsers-dev=${SONIC_TENSORRT_VERSION}"
ln -sfn "${SYSROOT}/usr/bin/tmux" "${BIN_DIR}/tmux"
# deploy.sh checks for clang although the already-built runtime does not invoke
# it. Keep the launcher non-root by exposing the working system C++ compiler.
if ! command -v clang >/dev/null 2>&1; then
    ln -sfn "$(command -v g++)" "${BIN_DIR}/clang"
fi

if [[ ! -x "${BIN_DIR}/just" ]]; then
    just_archive="${DOWNLOAD_DIR}/just-${SONIC_JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz"
    if [[ ! -f "${just_archive}" ]]; then
        curl --fail --location --retry 3 \
            "https://github.com/casey/just/releases/download/${SONIC_JUST_VERSION}/just-${SONIC_JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
            --output "${just_archive}.part"
        mv "${just_archive}.part" "${just_archive}"
    fi
    tar -xzf "${just_archive}" -C "${BIN_DIR}" just
fi

if [[ ! -f "${ONNX_DIR}/lib/libonnxruntime.so" ]]; then
    onnx_archive="${DOWNLOAD_DIR}/onnxruntime-linux-x64-${SONIC_ONNXRUNTIME_VERSION}.tgz"
    if [[ ! -f "${onnx_archive}" ]]; then
        curl --fail --location --retry 3 \
            "https://github.com/microsoft/onnxruntime/releases/download/v${SONIC_ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${SONIC_ONNXRUNTIME_VERSION}.tgz" \
            --output "${onnx_archive}.part"
        mv "${onnx_archive}.part" "${onnx_archive}"
    fi
    tar -xzf "${onnx_archive}" -C "${TOOLS_DIR}"
    ln -sfn "${TOOLS_DIR}/onnxruntime-linux-x64-${SONIC_ONNXRUNTIME_VERSION}" "${ONNX_DIR}"
fi

export PATH="${BIN_DIR}:${PATH}"
export onnxruntime_ROOT="${ONNX_DIR}"
export LD_LIBRARY_PATH="${ONNX_DIR}/lib:${SYSROOT}/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CPATH="${SYSROOT}/usr/include${CPATH:+:${CPATH}}"

cmake -S "${DEPLOY_DIR}" -B "${DEPLOY_DIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH="${SYSROOT}/usr;${ONNX_DIR}" \
    -DCMAKE_INCLUDE_PATH="${SYSROOT}/usr/include;${ONNX_DIR}/include" \
    -DCMAKE_LIBRARY_PATH="${SYSROOT}/usr/lib/x86_64-linux-gnu;${ONNX_DIR}/lib" \
    -DTensorRT_FIND_COMPONENTS="nvinfer;nvinfer_plugin;nvonnxparser" \
    -DTensorRT_INCLUDE_DIR="${SYSROOT}/usr/include/x86_64-linux-gnu" \
    -DTensorRT_nvinfer_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer.so.10 \
    -DTensorRT_nvinfer_plugin_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10 \
    -DTensorRT_nvonnxparser_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 \
    -DMSGPACK_INCLUDE_DIR="${SYSROOT}/usr/include" \
    -DZMQ_INCLUDE_DIR="${SYSROOT}/usr/include" \
    -DZMQ_LIBRARY="${SYSROOT}/usr/lib/x86_64-linux-gnu/libzmq.so"
cmake --build "${DEPLOY_DIR}/build" --parallel "$(nproc)"

[[ -x "${DEPLOY_DIR}/target/release/g1_deploy_onnx_ref" ]] || {
    echo "SONIC deploy build completed without the expected executable." >&2
    exit 1
}

echo "SONIC deploy is ready: ${DEPLOY_DIR}/target/release/g1_deploy_onnx_ref"
echo "Local tools are under: ${TOOLS_DIR}"
