#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

echo "=== 静态核对 fall_detection 算法包 ==="
cd "$SCRIPT_DIR"

pass() { echo "✓ $1"; }
fail() { echo "✗ $1"; exit 1; }

[ -f algo_meta.yaml ] || fail "缺少 algo_meta.yaml"
[ -f label_map.json ] || fail "缺少 label_map.json"
[ -f testimage.jpg ] || fail "缺少 testimage.jpg"
[ -f CMakeLists.txt ] || fail "缺少 CMakeLists.txt"
[ -f src/fall_detection.cpp ] || fail "缺少 src/fall_detection.cpp"
pass "必需源文件存在"

grep -q 'algorithm_name: "fall_detection"' algo_meta.yaml || fail "algorithm_name 不正确"
grep -q 'algorithm: "fall_detection"' algo_meta.yaml || fail "algorithm 不正确"
grep -q 'version: "1.0.0"' algo_meta.yaml || fail "version 不正确"
grep -q 'result_schema: "fall_pose_event"' algo_meta.yaml || fail "result_schema 不正确"
grep -q 'estimate_pose' algo_meta.yaml || fail "capabilities 缺少 estimate_pose"
grep -q 'YOLOv8-pose' algo_meta.yaml || fail "description 未声明 YOLOv8-pose"
for key in model_path input_size conf_thres iou_thres keypoint_thres fall_aspect_ratio torso_tilt_degree min_pose_points suspected_score fall_score max_detections; do
  grep -q "^[[:space:]]*$key:" algo_meta.yaml || fail "ai_params_schema 缺少 $key"
done
pass "algo_meta.yaml 参数完整"

if [ ! -x "$VENV_PYTHON" ]; then
  fail "未找到项目虚拟环境 Python: $VENV_PYTHON，请在项目根目录执行 uv venv .venv"
fi

"$VENV_PYTHON" <<'PY'
import json
from pathlib import Path
items = json.loads(Path('label_map.json').read_text())
assert isinstance(items, list), 'label_map 必须是数组'
codes = {item['category_code'] for item in items}
for code in (21001, 21002, 21003):
    assert code in codes, f'缺少 category_code {code}'
for item in items:
    code = item['category_code']
    assert isinstance(code, int) and 10000 <= code <= 99999, f'非法 category_code {code}'
print('✓ label_map.json 合法')
PY

for symbol in detector_init detector_infer detector_free_result detector_destroy detector_self_test detector_version detector_name; do
  grep -q "$symbol" src/fall_detection.cpp || fail "源码缺少 $symbol"
done
pass "ABI 函数实现存在"

grep -q 'onnxruntime_cxx_api.h' src/fall_detection.cpp || fail "未引入 ONNXRuntime"
grep -q 'opencv2/opencv.hpp' src/fall_detection.cpp || fail "未引入 OpenCV"
grep -q 'DecodeOutput' src/fall_detection.cpp || fail "缺少 YOLOv8-pose 输出解析"
grep -q 'AnalyzeFall' src/fall_detection.cpp || fail "缺少摔倒姿态判断"
grep -q 'kKeypointCount = 17' src/fall_detection.cpp || fail "未按 COCO 17 点解析"
grep -q 'torso_tilt_degree' src/fall_detection.cpp || fail "缺少躯干倾角判断"
pass "YOLOv8-pose 推理与姿态判断逻辑存在"

for script in download_models.sh build.sh validate.sh test.sh package.sh; do
  [ -x "$script" ] || fail "$script 不可执行"
  bash -n "$script" || fail "$script 语法错误"
done
pass "脚本可执行且语法正确"

echo "=== 静态核对通过 ==="
