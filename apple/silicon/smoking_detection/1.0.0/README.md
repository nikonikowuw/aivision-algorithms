# Apple Silicon 吸烟检测算法 POC

基于 DAMO-YOLO 人体与香烟模型的 Apple Silicon 吸烟检测算法包。

## 概述

在单路 4K 广角监控视频中：
1. 检测并跟踪人物
2. 在人物上部 ROI 区域检测香烟
3. 通过 5 次中 3 次的时序确认输出一次性告警事件

## 技术栈

- **推理后端**: CoreML (原生) / ONNX Runtime CoreML EP (兜底)
- **预处理**: Metal GPU (NV12/BGRA → RGB + LetterBox)
- **跟踪**: ByteTrack 风格 IoU 跟踪
- **ABI**: C ABI，遵循 `engine/include/algo/abi_contract.h`

## 构建

```bash
# macOS
make build

# 运行测试
make test

# 验证包
make validate

# 打包
make package
```

## 配置

支持通过 `detector_init(config_json)` 配置以下参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `backend` | `coreml_native` | 推理后端 |
| `analysis_fps` | `4` | 每秒分析帧数 |
| `max_persons` | `10` | 最大有效人物数 |
| `person_conf_threshold` | `0.50` | 人体置信度阈值 |
| `cigarette_conf_threshold` | `0.60` | 香烟置信度阈值 |
| `tile_enabled` | `true` | 分块检测开关 |
| `temporal_window` | `5` | 时序窗口大小 |
| `confirm_hits` | `3` | 确认所需最少 hit 次数 |
| `rearm_ms` | `2000` | 重新布防时间 |

完整参数见 `algo_meta.yaml`。

## 输出格式

检测结果为 JSON 数组，无新事件时返回空数组 `[]`。

```json
[
  {
    "category_code": 14001,
    "detect_confidence": 0.84,
    "bbox": [0.12, 0.18, 0.16, 0.62],
    "cigarette_bbox": [0.205, 0.265, 0.008, 0.006],
    "track_id": 7,
    "event_id": "smoking:a13f:42"
  }
]
```

## 模型来源

- 人体检测: ModelScope `iic/cv_tinynas_human-detection_damoyolo`
- 香烟检测: ModelScope `iic/cv_tinynas_object-detection_damoyolo_cigarette`

**注意**: ModelScope 权重许可证未明确，本包仅限内部 POC。

## 限制

- 首版将手持香烟视为吸烟，不验证嘴部动作
- 远距离香烟像素不足时无法恢复信息
- 简化 IoU tracker 在高密度交叉场景可能 ID switch
