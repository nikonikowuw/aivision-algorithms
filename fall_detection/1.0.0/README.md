# fall_detection 1.0.0

基于 **YOLOv8-pose** 的摔倒检测算法包，符合平台算法包规范与 C ABI 契约。

## 功能

- 使用 `models/yolov8n-pose.onnx` 执行人体检测与 17 点姿态估计。
- 解析 YOLOv8-pose 输出 `[1, 56, 8400]` 或 `[1, 8400, 56]`。
- 基于以下特征综合判断摔倒：
  - 人体检测框宽高比 `aspect_ratio`。
  - 肩-髋躯干中线相对竖直方向的倾角 `torso_tilt_degree`。
  - 鼻子与脚踝垂直跨度。
  - 有效关键点数量。
- 输出 `category_code` 与 `label_map.json` 保持一致。

## 目录结构

```text
fall_detection/1.0.0/
├── algo_meta.yaml
├── label_map.json
├── testimage.jpg
├── nikoniko_detector.so        # build.sh 生成
├── models/
│   └── yolov8n-pose.onnx       # download_models.sh 生成或手动放置
├── src/
│   └── fall_detection.cpp
├── tests/
│   └── test_abi.cpp
├── CMakeLists.txt
├── download_models.sh
├── build.sh
├── validate.sh
├── test.sh
└── package.sh
```

## 输出类别

| category_code | label | 说明 |
|---:|---|---|
| 21001 | `fall` | 摔倒 |
| 21002 | `suspected_fall` | 疑似摔倒 |
| 21003 | `normal_person` | 正常人员 |

## 输出 JSON 示例

```json
[
  {
    "category_code": 21001,
    "label": "fall",
    "detect_confidence": 0.9123,
    "fall_score": 0.8000,
    "aspect_ratio": 1.4800,
    "torso_tilt_degree": 72.5000,
    "valid_keypoints": 12,
    "bbox": { "x": 120.0, "y": 230.0, "w": 210.0, "h": 140.0 },
    "keypoints": [
      { "x": 180.0, "y": 250.0, "score": 0.88 }
    ]
  }
]
```

## 构建

依赖：

- CMake >= 3.16
- C++17 compiler
- OpenCV >= 4
- ONNXRuntime C++
- `algorithms/.venv` Python（由 `uv` 管理，仅用于自动导出模型）

```bash
cd algorithms/fall_detection/1.0.0
./download_models.sh
./build.sh
./test.sh
./package.sh
```

如果部署环境无法联网，可在 `algorithms/.venv` 中预先安装依赖并手动导出模型：

```bash
# 在 algorithms/ 目录执行
uv venv .venv
uv pip install --python .venv/bin/python ultralytics

cd fall_detection/1.0.0
../../.venv/bin/python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolov8n-pose.pt')
model.export(format='onnx', opset=12, simplify=True, dynamic=False, imgsz=640)
PY
mkdir -p models
mv yolov8n-pose.onnx models/yolov8n-pose.onnx
```

## ABI

导出符号：

- `detector_init`
- `detector_infer`
- `detector_free_result`
- `detector_destroy`
- `detector_self_test`
- `detector_version`
- `detector_name`

所有 C ABI 出口均捕获异常，禁止异常跨 ABI 边界传播。

## 参数说明

参数均在 `algo_meta.yaml` 的 `ai_params_schema` 中定义：

- `model_path`: YOLOv8-pose ONNX 模型路径。
- `input_size`: 模型输入尺寸，默认 `640`。
- `conf_thres`: 人体检测置信度阈值。
- `iou_thres`: NMS IOU 阈值。
- `keypoint_thres`: 姿态关键点可见性阈值。
- `fall_aspect_ratio`: 人体框横向宽高比阈值。
- `torso_tilt_degree`: 躯干横倒倾角阈值。
- `min_pose_points`: 最少有效关键点数。
- `suspected_score`: 疑似摔倒分数阈值。
- `fall_score`: 摔倒分数阈值。
- `max_detections`: 单帧最大输出人数。
