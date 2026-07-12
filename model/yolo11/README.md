# YOLO11 模型管理与转换工具 (YOLO11 Model Management & Export Tools)

本目录包含 YOLO11 官方模型的权重下载脚本以及将 PyTorch 格式（`.pt`）模型导出为各种平台推理格式（如 ONNX、CoreML、TensorRT 等）的优化导出脚本。

---

## 1. 输入与输出规格

| 规格类型 | 配置详情 | 说明 |
| :--- | :--- | :--- |
| **输入形状** | `(1, 3, 640, 640)` | BatchSize=1，三通道。高度与宽度在运行时可根据推理配置动态缩放。 |
| **通道顺序** | **RGB** | 图像预处理时需执行 BGR 到 RGB 的通道交换。 |
| **归一化方式** | 像素值除以 `255.0` | 映射至 `[0.0, 1.0]` 区间。无特殊的 mean/std 偏置。 |
| **输出张量** | 1 个输出，通常命名为 `output0` | 格式为 `(1, 4 + num_classes, 8400)`。 |
| **输出物理含义** | 坐标框与置信度展平矩阵 | - **前 4 维**：预测边界框的中心坐标及宽高 `(cx, cy, w, h)`。<br>- **后 `num_classes` 维**：各类别的分类置信度得分。<br>- **`8400`**：三个金字塔特征层展平后的 Anchor 预测总数（$80\times80 + 40\times40 + 20\times20$）。 |

---## 4. 权重下载

使用 `weights/download.sh` 脚本可一键下载 YOLO11 官方的 PyTorch 预训练权重。

### 4.1 用法

```bash
./weights/download.sh [n|s|m]
```

### 4.2 参数说明

* `n` (默认)：下载 YOLO11n (`yolo11n.pt`)，适用于低算力边缘端设备。
* `s`：下载 YOLO11s (`yolo11s.pt`)。
* `m`：下载 YOLO11m (`yolo11m.pt`)。

### 4.3 示例

```bash
# 下载默认 of nano 尺寸权重
./weights/download.sh n
```

---

## 5. 模型导出

使用 `python/export.py` 可以将下载好的 PyTorch 格式模型转换为其它推理格式。脚本已经过优化，支持全方位的参数控制。

### 5.1 核心命令行参数

| 参数名 | 简写 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `--weights` | `-w` | `str` | `yolo11n.pt` | 输入的 PyTorch 权重 file 路径。 |
| `--format` | `-f` | `str` | `onnx` | 导出格式。支持: `onnx`, `torchscript`, `openvino`, `engine`, `coreml`, `tflite` 等。 |
| `--imgsz` | `--img-size` | `int` | `640` | 输入尺寸。支持单值（如 `640`）或多值（如 `640 480`）。 |
| `--half` | - | `bool` | `False` | 启用 FP16 半精度模式导出。 |
| `--dynamic` | - | `bool` | `False` | 启用动态输入维度（适用于 ONNX/TensorRT）。 |
| `--opset` | - | `int` | `None` | 指定 ONNX Opset 版本号。 |
| `--int8` | - | `bool` | `False` | 启用 INT8 量化。 |
| `--batch` | - | `int` | `1` | 导出模型的 batch size。 |
| `--device` | - | `str` | `None` | 指定执行设备（如 `cpu` 或 `cuda:0`）。 |

### 5.2 导出示例

1. **导出为标准 ONNX 格式**：

    ```bash
    python python/export.py -w weights/yolo11n.pt -f onnx
    ```

2. **导出为带动态维度的 ONNX 格式**：

    ```bash
    python python/export.py -w weights/yolo11n.pt -f onnx --dynamic
    ```

3. **针对 Apple M 系列芯片导出为 CoreML 格式**：

    ```bash
    python python/export.py -w weights/yolo11n.pt -f coreml --half
    ```

4. **导出为 TensorRT Engine 格式（需在有 GPU 及 TensorRT 环境的主机上运行）**：

    ```bash
    python python/export.py -w weights/yolo11n.pt -f engine --half --device cuda:0
    ```

---

## 6. 平台专用转换工具 (conver.py)

针对 AIVisionInference 平台各边缘计算硬件节点的运行环境，提供了专用编译/转换工具 `python/conver.py`。该脚本负责将常规的 `.pt`/`.onnx` 转换为专用的加速模型格式。

### 6.1 命令行参数说明

| 参数名 | 简写 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | `str` | `weights/yolo11n.pt` | 输入模型文件路径（支持 `.pt` 或 `.onnx`）。若为 `.pt`，在转换 rknn/om 时会先自动导出为 `.onnx`。 |
| `--target` | `-t` | `str` | **(必填)** | 目标转换平台格式。支持：`coreml`, `rknn`, `om`。 |
| `--output` | `-o` | `str` | `None` | 指定保存的目标文件/目录路径。默认自动生成。 |
| `--soc` | - | `str` | `None` | 指定目标 SoC 芯片型号。针对 rknn 默认为 `rk3588`；针对 om 默认为 `Ascend310P3`。 |
| `--quantize` | - | `bool` | `False` | 启用 NPU INT8 量化编译（仅 rknn 可用）。 |
| `--dataset` | - | `str` | `None` | INT8 量化校准数据集路径（txt 文件，内含校准图片路径）。 |
| `--imgsz` | - | `int` | `640` | 模型输入图像大小。 |

### 6.2 平台转换示例

1. **编译为 Apple Silicon M 系列芯片专用的 CoreML 格式 (`.mlpackage`)**：

    ```bash
    python python/conver.py -i weights/yolo11n.pt -t coreml
    ```

2. **在 x86_64 开发机上使用 RKNN-Toolkit2 编译为瑞芯微 RK3588 NPU 专用的 RKNN 格式（不量化）**：

    ```bash
    python python/conver.py -i weights/yolo11n.pt -t rknn --soc rk3588
    ```

3. **在 RK3588 板端/开发机上启用 INT8 量化编译 RKNN 模型**：

    ```bash
    python python/conver.py -i weights/yolo11n.onnx -t rknn --soc rk3588 --quantize --dataset dataset.txt
    ```

4. **构建华为 Ascend 昇腾 ATC 编译指令（输出 `.om` 模型文件）**：

    ```bash
    python python/conver.py -i weights/yolo11n.onnx -t om --soc Ascend310P3
    ```

    *注：若系统中存在 `atc` 命令，将自动执行编译；若不存在，将打印 ATC 运行命令并提示您在 Ascend 开发环境主机上执行。*
