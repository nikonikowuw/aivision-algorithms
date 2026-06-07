#!/bin/bash

set -euo pipefail

echo "=== 打包 YOLOv8-pose 摔倒检测算法包 ==="

./validate.sh

binary_info="$(file -b nikoniko_detector.so 2>/dev/null || true)"
if printf '%s\n' "$binary_info" | grep -q 'Mach-O'; then
  echo "✗ 当前 nikoniko_detector.so 是 macOS Mach-O 产物，不能用于 Linux/RK Engine 部署"
  echo "  请在目标 Linux/RK 构建环境重新执行 ./build.sh 后再打包"
  exit 1
fi
if ! printf '%s\n' "$binary_info" | grep -q 'ELF'; then
  echo "✗ 无法确认 nikoniko_detector.so 是 Linux ELF 共享库: $binary_info"
  exit 1
fi

package_name="fall_detection_1.0.0"
package_dir="/tmp/${package_name}"
rm -rf "$package_dir"
mkdir -p "$package_dir"

cp algo_meta.yaml "$package_dir/"
cp label_map.json "$package_dir/"
cp testimage.jpg "$package_dir/"
cp nikoniko_detector.so "$package_dir/"
cp README.md "$package_dir/"
cp -r models "$package_dir/"

(
  cd /tmp
  rm -f "${package_name}.zip"
  zip -r "${package_name}.zip" "$package_name"
)

mv "/tmp/${package_name}.zip" ./
rm -rf "$package_dir"

echo "✓ 输出文件: ${package_name}.zip"
echo "=== 打包完成 ==="
