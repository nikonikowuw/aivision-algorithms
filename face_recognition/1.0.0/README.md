# 人脸识别算法包

基于 YOLOv11n、ByteTrack、YOLOv11n-face、2d106det 和 w600k_r50 的人脸识别算法包。

## 功能

- **人体检测**: 使用 `models/yolov11n.onnx` 检测人体区域
- **人体追踪**: 使用 ByteTrack 输出稳定 `track_id`
- **人脸检测**: 使用 `models/yolov11n-face.onnx` 检测人脸
- **人脸对齐**: 使用 `models/insightface/2d106det.onnx` 加载对齐模型，并按关键点执行 112x112 仿射对齐
- **人脸识别**: 使用 `models/insightface/w600k_r50.onnx` 提取 512 维特征并执行 1:N 比对

## 必需文件

```text
face_recognition/1.0.0/
├── algo_meta.yaml
├── label_map.json
├── testimage.jpg
├── nikoniko_detector.so      # make 后生成
├── models/
│   ├── yolov11n.onnx         # 人体检测 (CoreML MLProgram)
│   ├── yolov11n-face.onnx    # 人脸检测 (CoreML MLProgram)
│   ├── yolov11n.mlpackage    # CoreML 原生格式 (可选)
│   ├── yolo11n.pt            # Ultralytics 原始模型
│   └── insightface/
│       ├── 2d106det.onnx     # 人脸对齐 (CoreML NeuralNetwork)
│       └── w600k_r50.onnx    # 人脸识别 (CoreML NeuralNetwork)
├── CMakeLists.txt
├── Makefile
├── include/
│   ├── face_recognition.h
│   └── ort_coreml.h
└── src/
    ├── face_recognition.cpp
    ├── insightface_recognizer.cpp
    ├── yolov11_detector.cpp
    └── byte_tracker.cpp
```

## 构建

```bash
make
```

校验 ABI 符号：

```bash
make symbols
```

打包：

```bash
make package
```

清理：

```bash
make distclean
```

## YOLOv11n ONNX 导出

当前包已从 `models/yolo11n.pt` 导出 `models/yolov11n.onnx`。如需重新导出：

```bash
python3 - <<'PY'
from pathlib import Path
from ultralytics import YOLO
model = YOLO('models/yolo11n.pt')
exported = Path(model.export(format='onnx', imgsz=640, opset=12, simplify=False))
target = Path('models/yolov11n.onnx')
if exported != target:
    target.unlink(missing_ok=True)
    exported.replace(target)
print(target)
PY
```

## Mac CoreML 策略

模型推理强制使用 ONNXRuntime CoreML EP，InsightFace 模型禁用 CPU EP fallback。只要 InsightFace 模型有节点不能分配给 CoreML，初始化会直接失败，算法不可用。

当前模型 CoreML 状态：

| 模型 | CoreML 格式 | 状态 | 备注 |
| --- | --- | --- | --- |
| `yolov11n-face.onnx` | MLProgram | ✅ 100% CoreML | float16 自动处理 |
| `2d106det.onnx` | NeuralNetwork | ✅ 100% CoreML | float32 |
| `w600k_r50.onnx` | NeuralNetwork | ✅ 100% CoreML | 融合末端BN+Reshape |
| `yolov11n.onnx` | MLProgram | ⚠️ 311/321 CoreML | 10个Split节点被ORT放CPU（设计行为） |

YOLO 检测模型可导出 CoreML 格式用于后续原生 CoreML 路径。当前已生成 `models/yolov11n.mlpackage`。如需重新导出：

```bash
make yolo-coreml
```

如果 `models/yolo11n.pt` 不存在，脚本会使用 Ultralytics 自动下载官方 `yolo11n.pt`；`models/yolov11n.onnx` 不直接用于 CoreML 导出。

InsightFace ONNX 可用 CoreML-only 验证脚本检查是否存在 CPU fallback：

```bash
make setup-mac-ml
make verify-insightface-coreml
```

验证脚本只注册 `CoreMLExecutionProvider`，并设置 `session.disable_cpu_ep_fallback=1`。如果 InsightFace 模型需要 CPU fallback，会直接失败。

## 配置参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `person_conf_thres` | `0.5` | YOLOv11n 人体检测置信度阈值 |
| `iou_thres` | `0.45` | NMS IOU 阈值 |
| `enable_tracker` | `true` | 是否启用 ByteTrack |
| `face_conf_thres` | `0.6` | YOLOv11n-face 人脸检测置信度阈值 |
| `face_quality_thres` | `0.3` | 人脸质量阈值 |
| `max_faces` | `10` | 单帧最大人脸数 |
| `recognition_threshold` | `0.6` | 人脸识别相似度阈值 |

## 输出示例

```json
{
  "results": [
    {
      "category_code": 13001,
      "identity_id": "employee_001",
      "similarity": 0.923,
      "detect_confidence": 0.981,
      "bbox": { "x": 120, "y": 150, "w": 40, "h": 40 },
      "landmarks": [
        { "x": 130, "y": 160 },
        { "x": 145, "y": 160 },
        { "x": 137, "y": 170 }
      ],
      "track_id": 1,
      "quality_score": 0.85
    }
  ],
  "face_count": 1,
  "frame_width": 1920,
  "frame_height": 1080,
  "infer_time_us": 42000
}
```

## 类别编码

| category_code | 说明 |
| --- | --- |
| `13001` | 已识别人员 |
| `13002` | 未知人员 |
| `13003` | 人员 |
