#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

ensure_venv_python() {
  if [ -x "$VENV_PYTHON" ]; then
    return 0
  fi
  echo "✗ 未找到项目虚拟环境 Python: $VENV_PYTHON"
  echo "  请在项目根目录执行: uv venv .venv"
  exit 1
}

ensure_python_package() {
  local module_name="$1"
  local package_name="$2"
  if ! "$VENV_PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('$module_name') else 1)"; then
    if ! command -v uv >/dev/null 2>&1; then
      echo "✗ 未找到 uv，不能向项目 .venv 安装 $package_name"
      exit 1
    fi
    echo "安装 $package_name 到项目 .venv..."
    uv pip install --python "$VENV_PYTHON" "$package_name"
  fi
}

echo "=== 验证 YOLOv8-pose 摔倒检测算法包 ==="
cd "$SCRIPT_DIR"

required_files=(
  "algo_meta.yaml"
  "label_map.json"
  "testimage.jpg"
  "CMakeLists.txt"
  "src/fall_detection.cpp"
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "✗ 缺少必需文件: $file"
    exit 1
  fi
  echo "✓ $file"
done

if [ ! -f "models/yolov8n-pose.onnx" ]; then
  echo "✗ 缺少模型: models/yolov8n-pose.onnx"
  echo "  请执行 ./download_models.sh 或手动放置 YOLOv8-pose ONNX 模型"
  exit 1
fi
echo "✓ models/yolov8n-pose.onnx"

if [ ! -f "nikoniko_detector.so" ]; then
  echo "✗ 缺少动态库: nikoniko_detector.so，请先执行 ./build.sh"
  exit 1
fi
echo "✓ nikoniko_detector.so"

if command -v nm >/dev/null 2>&1; then
  if [ "$(uname -s)" = "Darwin" ]; then
    nm_output="$(nm -gU nikoniko_detector.so 2>/dev/null || true)"
  else
    nm_output="$(nm -D nikoniko_detector.so 2>/dev/null || nm -g nikoniko_detector.so 2>/dev/null || true)"
  fi

  for symbol in detector_init detector_infer detector_free_result detector_destroy detector_self_test detector_version detector_name; do
    if printf '%s\n' "$nm_output" | grep -Eq "(^|[[:space:]_])${symbol}$"; then
      echo "✓ symbol: $symbol"
    else
      echo "✗ 缺少符号: $symbol"
      exit 1
    fi
  done
else
  echo "○ 未找到 nm，跳过符号检查"
fi

ensure_venv_python
ensure_python_package "yaml" "pyyaml"

"$VENV_PYTHON" <<'PY'
import json
from pathlib import Path
import yaml

meta = yaml.safe_load(Path('algo_meta.yaml').read_text())
for field in ['algorithm_name', 'algorithm', 'version', 'domain', 'result_schema', 'capabilities', 'ai_params_schema']:
    if field not in meta:
        raise SystemExit(f'缺少 algo_meta 字段: {field}')
if meta['algorithm_name'] != 'fall_detection':
    raise SystemExit('algorithm_name 必须为 fall_detection')
if meta['algorithm'] != 'fall_detection':
    raise SystemExit('algorithm 必须为 fall_detection')
props = meta['ai_params_schema'].get('properties', {})
for key in ['model_path', 'input_size', 'conf_thres', 'iou_thres', 'keypoint_thres', 'fall_aspect_ratio', 'torso_tilt_degree']:
    if key not in props:
        raise SystemExit(f'缺少参数定义: {key}')

label_map = json.loads(Path('label_map.json').read_text())
category_codes = {item.get('category_code') for item in label_map}
for code in [21001, 21002, 21003]:
    if code not in category_codes:
        raise SystemExit(f'缺少 category_code: {code}')
for item in label_map:
    code = item.get('category_code')
    if not isinstance(code, int) or not 10000 <= code <= 99999:
        raise SystemExit(f'category_code 非法: {code}')
print('✓ YAML/JSON 内容校验通过')
PY

echo "=== 验证通过 ==="
