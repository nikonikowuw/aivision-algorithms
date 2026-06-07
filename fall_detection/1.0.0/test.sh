#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "=== 测试 YOLOv8-pose 摔倒检测算法包 ==="
cd "$SCRIPT_DIR"

./validate.sh

echo ""
echo "=== 编译 ABI 测试程序 ==="
abi_compile_args=(-std=c++17 tests/test_abi.cpp -ldl -o /tmp/fall_detection_test_abi)
if [ "$(uname -s)" = "Darwin" ] && command -v xcrun >/dev/null 2>&1; then
  sdkroot="$(xcrun --show-sdk-path)"
  if [ -n "$sdkroot" ]; then
    abi_compile_args=(-isysroot "$sdkroot" -isystem "$sdkroot/usr/include/c++/v1" "${abi_compile_args[@]}")
  fi
fi
g++ "${abi_compile_args[@]}"
/tmp/fall_detection_test_abi ./nikoniko_detector.so

echo ""
echo "=== 执行算法自检 ==="
if [ ! -x "$VENV_PYTHON" ]; then
  echo "✗ 未找到项目虚拟环境 Python: $VENV_PYTHON"
  echo "  请在项目根目录执行: uv venv .venv"
  exit 1
fi

"$VENV_PYTHON" <<'PY'
import ctypes
import os

lib = ctypes.CDLL(os.path.abspath('./nikoniko_detector.so'))
lib.detector_self_test.restype = ctypes.c_int
ret = lib.detector_self_test()
if ret != 0:
    raise SystemExit(f'✗ detector_self_test 失败: {ret}')
print('✓ detector_self_test 通过')
PY

echo "=== 测试完成 ==="
