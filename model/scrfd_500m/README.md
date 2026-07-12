# SCRFD 人脸检测模型管理与转换工具 (SCRFD Face Detection Model Management & Export Tools)

本目录包含 SCRFD 500M (轻量高效人脸检测器) 模型的导出与平台转换脚本，可将模型转换为各种硬件平台加速推理格式（ONNX、CoreML、RKNN、OM）。

---

## 1. 输入与输出规格

| 规格类型 | 配置详情 | 说明 |
| :--- | :--- | :--- |
| **输入形状** | `(1, 3, 640, 640)` | BatchSize=1，三通道。支持动态宽高输入。 |
| **通道顺序** | **RGB** | 预处理需执行 BGR 到 RGB 通道交换。 |
| **归一化方式** | `(x - 127.5) / 127.5` | 映射至 `[-1.0, 1.0]`，均值 127.5，标准差 127.5。 |
| **输出张量** | **6 个** (不带关键点) 或 **9 个** (带关键点 kps) | 按照不同下采样步长 (Stride 8, 16, 32) 分离输出。 |
| **输出物理含义** | 多尺度分类得分、边框偏置及人脸五点 | - **Scores (3个)**: `(1, 12800, 1)`, `(1, 3200, 1)`, `(1, 800, 1)`（代表各层 Anchor 包含人脸的置信度）。<br>- **Bboxes (3个)**: `(1, 12800, 4)`, `(1, 3200, 4)`, `(1, 800, 4)`（边界框相对于 Anchor 中心的 `(left, top, right, bottom)` 偏移）。<br>- **Kps (3个, 仅限 KPS 版)**: `(1, 12800, 10)`, `(1, 3200, 10)`, `(1, 800, 10)`（5 个面部关键点 `(x, y)` 偏移量）。 |

---

## 2. 目录结构

```text
scrfd/
├── weights/
│   ├── scrfd_500m.pth                 # 原始 PyTorch 训练权重 (MMDetection/MMCV 格式)
│   └── download.sh                    # ONNX 权重一键下载脚本
├── python/
│   ├── export.py                      # 核心导出脚本 (将 .pth 编译为 .onnx)
│   └── conver.py                      # 多平台 (CoreML, RKNN, Ascend OM) 专用转换编译脚本
└── README.md                          # 本说明文档
```

---

## 3. 环境准备

确保已安装以下必要 Python 依赖包：

```bash
pip install torch coremltools onnx
```

---

## 4. 快速获取 ONNX 模型

如果您的本地环境（例如 macOS 平台）中缺少 `mmdet` 或 `mmcv` 的复杂 C++ 编译依赖，您可以直接下载 InsightFace 官方预编译的 `scrfd_500m.onnx` 模型，跳过从 `.pth` 的导出过程：

### 4.1 运行下载脚本
```bash
./weights/download.sh
```

### 4.2 或运行导出脚本带 `--download-onnx` 参数
```bash
python python/export.py --download-onnx
```
这两种方法都会直接从官方渠道下载 `scrfd_500m.onnx` 并保存到 `weights/` 目录中。

---

## 5. 从 PyTorch .pth 导出为 ONNX

如果您安装了 MMDetection 与 MMCV 编译环境，并希望从您自己训练的 `scrfd_500m.pth` 导出 ONNX 模型：

```bash
python python/export.py -w weights/scrfd_500m.pth -c <PATH_TO_CONFIG_PY> --shape 640 640
```
*注：`<PATH_TO_CONFIG_PY>` 是 MMDetection 对应的模型配置文件，例如官方库中的 `configs/scrfd/scrfd_500m.py`。*

---

## 6. 多平台转换编译 (conver.py)

利用 `conver.py` 可以将导出的 `scrfd_500m.onnx` 模型一键编译转换到特定的目标硬件加速器：

### 6.1 编译为 Apple Silicon M 系列芯片专用的 CoreML 格式 (`.mlpackage`)：
```bash
python python/conver.py -i weights/scrfd_500m.onnx -t coreml
```

### 6.2 编译为瑞芯微 RK3588 NPU 专用的 RKNN 格式（内置 RGB 均值与方差归一化）：
```bash
python python/conver.py -i weights/scrfd_500m.onnx -t rknn --soc rk3588
```
*注：`conver.py` 中已为您内置配置了 SCRFD RGB 标准归一化参数 `mean_values=[[127.5, 127.5, 127.5]]` 与 `std_values=[[127.5, 127.5, 127.5]]`。*

### 6.3 构建华为 Ascend 昇腾 ATC 编译指令（输出 `.om` 模型文件）：
```bash
python python/conver.py -i weights/scrfd_500m.onnx -t om --soc Ascend310P3 --imgsz 640 640
```
*注：若当前环境中已安装并配置 CANN ATC 编译器，该脚本会自动拉起 ATC 进行模型编译；否则它将打印完整的 ATC 转换指令供您在华为开发机上执行。*
