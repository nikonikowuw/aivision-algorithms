# SCRFD Person 人体检测模型管理与转换工具 (SCRFD Person Detection Model Management & Export Tools)

本目录包含 SCRFD Person 2.5G (人体全身体态检测器) 模型的管理与平台转换脚本，可将模型转换为各种硬件平台加速推理格式（ONNX、CoreML、RKNN、OM）。

---

## 1. 模型介绍

`scrfd_person_2.5g` 是由著名开源人脸分析项目 **InsightFace** 官方提供的高效**人体检测模型**（全身体态检测器）。

### 1.1 技术背景与设计理念

该模型继承了 **SCRFD** (Sample and Computation Redistribution for Efficient Face Detection) 核心设计：

* **计算重分配 (Computation Redistribution)**：借助神经架构搜索 (NAS)，将计算算力以最优比例重新分配于骨干网络 (Backbone)、特征融合网络 (Neck/PAFPN) 与检测头 (Head) 之间，使得轻量化模型也能具备优异的小目标（远距离人体）检测精度。
* **共享检测头 (Shared Head)**：模型的分类与边界框回归头在不同的特征金字塔层级（FPN levels）间共享参数，极大缩减了参数量，提高了边缘硬件上的运行效率与缓存命中率。

---

## 2. 输入与输出规格

| 规格类型 | 配置详情 | 说明 |
| :--- | :--- | :--- |
| **输入形状** | `(1, 3, 640, 640)` | BatchSize=1，三通道。高度与宽度在编译/推理时可动态调整（如 320x320 等）。 |
| **通道顺序** | **RGB** | 图像预处理时需执行 BGR 到 RGB 的通道交换。 |
| **归一化方式** | `(x - 127.5) / 127.5` | 映射至 `[-1.0, 1.0]`，均值 127.5，标准差 127.5。 |
| **输出张量** | **6 个** 输出张量 | 按照不同下采样步长 (Stride 8, 16, 32) 分离输出。结构上等同于不带关键点版本的 SCRFD 脸部检测模型。 |
| **输出物理含义** | 多尺度分类得分与人体全身框 | - **Scores (3个)**: `(1, 12800, 1)`, `(1, 3200, 1)`, `(1, 800, 1)`（各层 Anchor 包含人体的置信度）。<br>- **Bboxes (3个)**: `(1, 12800, 4)`, `(1, 3200, 4)`, `(1, 800, 4)`（人体边界框相对于 Anchor 中心的偏移）。 |

---

## 3. 目录结构

```text
scrfd-person/
├── weights/
│   ├── scrfd_person_2.5g.onnx         # 原始 ONNX 模型权重
│   └── download.sh                    # ONNX 权重一键下载脚本
├── python/
│   ├── export.py                      # 核心同步/验证脚本
│   └── conver.py                      # 多平台 (CoreML, RKNN, Ascend OM) 专用转换编译脚本
└── README.md                          # 本说明文档
```

## 4. 环境准备

确保已安装以下必要 Python 依赖包：

```bash
pip install torch coremltools onnx onnx2torch
```

---

## 5. 权重下载

如果本地缺失 `weights/scrfd_person_2.5g.onnx`，您可以直接拉取官方预训练版本：

### 5.1 运行下载脚本

```bash
./weights/download.sh
```

### 5.2 或运行同步脚本带 `--download-onnx` 参数

```bash
python python/export.py --download-onnx
```

---

## 6. 模型验证

运行 `python/export.py` 脚本来验证模型是否放置在正确的位置：

```bash
python python/export.py
```

---

## 7. 多平台转换编译 (conver.py)

利用 `conver.py` 将 `scrfd_person_2.5g.onnx` 编译转换到特定的目标硬件加速器：

### 7.1 编译为 Apple Silicon M 系列芯片专用的 CoreML 格式 (`.mlpackage`)

```bash
python python/conver.py -i weights/scrfd_person_2.5g.onnx -t coreml
```

### 7.2 编译为瑞芯微 RK3588 NPU 专用的 RKNN 格式（内置 RGB 均值与方差归一化）

```bash
python python/conver.py -i weights/scrfd_person_2.5g.onnx -t rknn --soc rk3588
```

*注：`conver.py` 中已配置 RGB 标准归一化参数 `mean_values=[[127.5, 127.5, 127.5]]` 与 `std_values=[[127.5, 127.5, 127.5]]`。*

### 7.3 构建华为 Ascend 昇腾 ATC 编译指令（输出 `.om` 模型文件）

```bash
python python/conver.py -i weights/scrfd_person_2.5g.onnx -t om --soc Ascend310P3 --imgsz 640 640
```

*注：若当前环境中已安装并配置 CANN ATC 编译器，该脚本会自动拉起 ATC 进行模型编译；否则它将打印完整的 ATC 转换指令供您在华为开发机上执行。*
