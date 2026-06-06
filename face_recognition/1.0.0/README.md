# 人脸识别算法包

基于InsightFace buffalo_l模型和YOLOv8-n的人脸识别算法包，支持ByteTracker人体追踪。

## 功能特性

- **人脸检测**: 使用InsightFace模型进行高精度人脸检测
- **人脸特征提取**: 提取512维人脸特征向量
- **人脸识别**: 1:N人脸识别，支持身份比对
- **人体检测**: 使用YOLOv8-n进行人体检测
- **人体追踪**: 可选ByteTracker进行人体追踪
- **质量评估**: 人脸质量评分和过滤

## 算法能力

| 能力 | 说明 |
|------|------|
| `detect` | 人脸检测 |
| `estimate_keypoints` | 人脸关键点检测 |
| `extract_embedding` | 人脸特征提取 |
| `track` | 人体追踪（ByteTracker） |
| `recognize_identity` | 身份识别 |

## 模型要求

### InsightFace模型
- `det_10g.onnx`: 人脸检测模型
- `w600k_r50.onnx`: 人脸特征提取模型

### YOLOv8模型
- `yolov8n.onnx`: 人体检测模型

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `conf_thres` | number | 0.5 | 人体检测置信度阈值 (0.0-1.0) |
| `iou_thres` | number | 0.45 | NMS IOU阈值 (0.0-1.0) |
| `enable_tracker` | boolean | true | 是否启用ByteTracker人体追踪 |
| `face_conf_thres` | number | 0.6 | 人脸检测置信度阈值 (0.0-1.0) |
| `face_quality_thres` | number | 0.3 | 人脸质量阈值 (0.0-1.0) |
| `max_faces` | integer | 10 | 单帧最大人脸数 (1-100) |
| `recognition_threshold` | number | 0.6 | 人脸识别相似度阈值 (0.0-1.0) |

## 输出格式

### 人脸识别结果

```json
[
  {
    "category_code": 13001,
    "identity_id": "employee_001",
    "similarity": 0.923,
    "detect_confidence": 0.981,
    "bbox": {
      "x": 120,
      "y": 150,
      "w": 40,
      "h": 40
    },
    "landmarks": [
      { "x": 130, "y": 160 },
      { "x": 145, "y": 160 },
      { "x": 137, "y": 170 }
    ],
    "track_id": 1,
    "quality_score": 0.85
  }
]
```

### 类别编码

| category_code | 说明 |
|---------------|------|
| 13001 | 已识别人员 |
| 13002 | 未知人员 |
| 13003 | 人员 |

## 目录结构

```
face_recognition/
├── algo_meta.yaml          # 算法元数据
├── nikoniko_detector.so    # 算法动态库
├── testimage.jpg           # 自检图片
├── label_map.json          # 类别映射
├── models/                 # 模型目录
│   ├── insightface/        # InsightFace模型
│   │   ├── det_10g.onnx
│   │   └── w600k_r50.onnx
│   ├── yolov8n.onnx        # YOLOv8模型
│   └── embeddings.json     # 已知人脸库（可选）
├── CMakeLists.txt          # 构建配置
├── README.md               # 说明文档
└── src/                    # 源代码
    ├── face_recognition.cpp
    ├── insightface_recognizer.cpp
    ├── yolov8_detector.cpp
    ├── byte_tracker.cpp
    └── utils.cpp
```

## 构建说明

### 依赖项

- OpenCV 4.x
- ONNXRuntime 1.x
- jsoncpp
- CMake 3.16+

### 编译

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### 安装

```bash
make install
```

## 使用说明

### 1. 准备模型文件

将InsightFace和YOLOv8模型文件放入`models/`目录。

### 2. 准备人脸库（可选）

如果有已知人脸库，创建`models/embeddings.json`文件：

```json
[
  {
    "id": "employee_001",
    "embedding": [0.1, 0.2, ...]  // 512维特征向量
  },
  {
    "id": "employee_002",
    "embedding": [0.3, 0.4, ...]
  }
]
```

### 3. 上传算法包

将整个目录打包为zip文件上传到平台。

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 人脸检测延迟 | ≤ 50ms |
| 人脸特征提取延迟 | ≤ 30ms |
| 人体检测延迟 | ≤ 40ms |
| 单帧推理总延迟 | ≤ 150ms |
| 人脸检测准确率 | ≥ 95% |
| 人脸识别准确率 | ≥ 90% |

## 注意事项

1. 模型文件需要从InsightFace官方仓库下载
2. YOLOv8模型需要从Ultralytics官方仓库下载
3. 首次运行需要初始化模型，可能需要较长时间
4. 建议在GPU环境下运行以获得最佳性能
5. 人脸库大小会影响识别速度，建议控制在10000人以内

## 自检机制

算法包实现了两层自检：

### 1. 平台侧自检（Go端执行）

算法包上传时，平台会自动执行以下检查：
- 文件校验：检查 `algo_meta.yaml`、`nikoniko_detector.so`、`testimage.jpg`、`label_map.json` 是否存在
- 动态库加载：通过 `dlopen` 加载 `.so` 文件
- 符号检查：验证 `detector_init`、`detector_infer`、`detector_destroy` 等符号是否存在
- 推理测试：使用 `testimage.jpg` 执行一次推理
- 结果校验：验证输出JSON格式和 `category_code` 是否符合规范

### 2. 算法包内部自检（`detector_self_test`）

算法包内部实现了 `detector_self_test` 可选符号，执行更详细的自检：

**自检流程：**
1. 读取并解析 `label_map.json`，收集有效的 `category_code`
2. 读取 `testimage.jpg` 测试图片
3. 初始化人脸检测器和人体检测器
4. 执行一次完整的推理流程
5. 验证输出格式：
   - 边界框有效性（宽度和高度 > 0）
   - 置信度范围（0.0 ~ 1.0）
   - 关键点数量（5个）
   - 特征向量维度（512维）
   - 特征向量归一化
6. 测试追踪器功能

**自检通过条件：**
- 动态库可成功加载
- 所有必需符号存在
- 模型可成功初始化（如果模型文件存在）
- 测试图片可成功读取和解码
- 推理结果格式正确
- `category_code` 在有效范围内

**自检不负责判断：**
- 检测框是否准确
- 分类标签是否正确
- 人脸身份是否匹配
- 置信度是否达到业务阈值
- 模型精度、召回率等质量指标

## 版本历史

- **1.0.0** (2026-06-06): 初始版本
  - 基于InsightFace buffalo_l的人脸识别
  - 基于YOLOv8-n的人体检测
  - 可选ByteTracker人体追踪
  - 支持动态参数配置
  - 实现完整的自检机制