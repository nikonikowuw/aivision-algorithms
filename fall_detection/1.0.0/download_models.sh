#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

ensure_venv_python() {
    if [ -x "$VENV_PYTHON" ]; then
        return 0
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "错误: 未找到 $VENV_PYTHON，且系统未安装 uv"
        echo "请在项目根目录执行: uv venv .venv"
        exit 1
    fi
    echo "未找到项目虚拟环境 Python，使用 uv 创建: $PROJECT_ROOT/.venv"
    uv venv "$PROJECT_ROOT/.venv"
}

ensure_python_package() {
    local module_name="$1"
    local package_name="$2"
    if ! "$VENV_PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('$module_name') else 1)"; then
        if ! command -v uv >/dev/null 2>&1; then
            echo "错误: 未找到 uv，不能向项目 .venv 安装 $package_name"
            exit 1
        fi
        echo "安装 $package_name 到项目 .venv..."
        uv pip install --python "$VENV_PYTHON" "$package_name"
    fi
}

echo "=== 下载 YOLOv8-pose 模型 ==="
cd "$SCRIPT_DIR"
mkdir -p models

if [ -f "models/yolov8n-pose.onnx" ]; then
    echo "✓ models/yolov8n-pose.onnx 已存在"
    exit 0
fi

ensure_venv_python
ensure_python_package "ultralytics" "ultralytics"

"$VENV_PYTHON" <<'PY'
from pathlib import Path
from ultralytics import YOLO

model_path = Path('models/yolov8n-pose.onnx')
model = YOLO('yolov8n-pose.pt')
model.export(format='onnx', opset=12, simplify=True, dynamic=False, imgsz=640)
exported = Path('yolov8n-pose.onnx')
if not exported.exists():
    raise SystemExit('导出失败: yolov8n-pose.onnx 不存在')
model_path.parent.mkdir(parents=True, exist_ok=True)
exported.replace(model_path)
print(f'✓ 已生成 {model_path}')
PY

echo "=== 模型准备完成 ==="
