# AdaFace 人脸识别模型管理与转换工具 (AdaFace Face Recognition Model Management & Export Tools)

本目录包含 AdaFace (基于 IR-101 骨干网络) 人脸识别模型的权重下载脚本，以及将 PyTorch checkpoint 导出和编译为各种硬件平台加速推理格式（ONNX、CoreML、RKNN、OM）的专用工具链。

---

## 1. 目录结构

```text
adaface/
├── weights/
│   ├── adaface_ir101_webface4m.ckpt   # 预训练 PyTorch 模型权重（约 650MB）
│   └── download.sh                    # 权重一键下载脚本
├── python/
│   ├── net.py                         # 官方 AdaFace 网络架构定义 (包含 GNAP, IR-101 等)
│   ├── export.py                      # 核心导出脚本 (将 .ckpt 编译为 .onnx 或 .mlpackage)
│   └── conver.py                      # 多平台 (CoreML, RKNN, Ascend OM) 专用转换编译脚本
└── README.md                          # 本说明文档
```

---

## 2. 环境准备

在执行导出和转换脚本之前，请确保已安装必要的 Python 依赖包。
建议使用 Python 虚拟环境，并安装以下包：

```bash
pip install torch torchvision coremltools transformers huggingface_hub
```

*注：确保 `numpy` 版本限制在 `<2.4.0`（例如 `numpy==2.3.5`），以防 NumPy 2.x 与 `coremltools` 产生类型转换冲突导致 CoreML 导出失败。*

---

## 3. 权重下载

使用 `weights/download.sh` 脚本可一键从 Hugging Face 下载 AdaFace IR-101 (WebFace4M) 的官方预训练模型权重。

### 3.1 用法

```bash
./weights/download.sh
```

---

## 4. 模型导出与多平台转换

通过 `python/export.py` 与 `python/conver.py` 可以方便地处理权重导出并编译为特定的运行环境。

### 4.1 导出为通用 ONNX / Mac Apple Silicon CoreML

1. **导出为通用 ONNX 格式 (`.onnx`)**：

    ```bash
    python python/export.py -w weights/adaface_ir101_webface4m.ckpt -f onnx --imgsz 112
    ```

2. **直接导出为 CoreML 格式 (`.mlpackage`)**：

    ```bash
    python python/export.py -w weights/adaface_ir101_webface4m.ckpt -f coreml --imgsz 112
    ```

### 4.2 平台专用编译 (conver.py)

1. **编译为 Apple Silicon M 系列芯片专用的 CoreML 格式**：

    ```bash
    python python/conver.py -i weights/adaface_ir101_webface4m.ckpt -t coreml
    ```

2. **编译为瑞芯微 RK3588 NPU 专用的 RKNN 格式（内置 mean/std 归一化对齐）**：

    ```bash
    python python/conver.py -i weights/adaface_ir101_webface4m.ckpt -t rknn --soc rk3588
    ```

    *注：AdaFace 输入像素范围为 BGR `[0, 255]`，模型归一化方式为减去 127.5 并除以 127.5。`conver.py` 中已为您内置配置 `mean_values=[[127.5, 127.5, 127.5]]` 与 `std_values=[[127.5, 127.5, 127.5]]`。*

3. **构建华为 Ascend 昇腾 ATC 编译指令（输出 `.om` 模型文件）**：

    ```bash
    python python/conver.py -i weights/adaface_ir101_webface4m.ckpt -t om --soc Ascend310P3
    ```

    *注：若当前环境中已安装并配置 CANN ATC 编译器，该脚本会自动拉起 ATC 进行模型编译；否则它将打印完整的 ATC 转换指令供您在华为开发机上执行。*
