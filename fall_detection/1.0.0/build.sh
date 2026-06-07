#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 构建摔倒检测算法包 ==="

if [ ! -f "models/yolov8n-pose.onnx" ]; then
    echo "未找到 models/yolov8n-pose.onnx，尝试下载/导出..."
    ./download_models.sh
fi

cmake_args=(
    -S .
    -B build
    -DCMAKE_BUILD_TYPE=Release
)

if [ "$(uname -s)" = "Darwin" ] && command -v xcrun >/dev/null 2>&1; then
    sdkroot="$(xcrun --show-sdk-path)"
    if [ -n "$sdkroot" ]; then
        cmake_args+=("-DCMAKE_OSX_SYSROOT=$sdkroot")
    fi
fi

if [ -z "${OpenCV_DIR:-}" ] && command -v brew >/dev/null 2>&1; then
    if brew list opencv >/dev/null 2>&1; then
        opencv_prefix="$(brew --prefix opencv)"
        if [ -d "$opencv_prefix/lib/cmake/opencv4" ]; then
            cmake_args+=("-DOpenCV_DIR=$opencv_prefix/lib/cmake/opencv4")
        fi
    fi
fi

if [ -z "${ONNXRuntime_ROOT:-}" ] && [ -z "${ONNXRUNTIME_ROOT:-}" ] && command -v brew >/dev/null 2>&1; then
    if brew list onnxruntime >/dev/null 2>&1; then
        onnxruntime_prefix="$(brew --prefix onnxruntime)"
        cmake_args+=("-DONNXRuntime_ROOT=$onnxruntime_prefix")
    else
        echo "✗ 未安装 ONNXRuntime C++ SDK"
        echo "  请先执行: brew install onnxruntime"
        echo "  或手动下载 ONNXRuntime SDK 后执行:"
        echo "  ONNXRuntime_ROOT=/path/to/onnxruntime ./build.sh"
        exit 1
    fi
fi

rm -rf build
mkdir -p build
cmake "${cmake_args[@]}"
cmake --build build --config Release -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cp build/nikoniko_detector.so ./nikoniko_detector.so

echo "✓ 输出: nikoniko_detector.so"
echo "=== 构建完成 ==="
