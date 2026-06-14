# 硬件适配补充说明：Rockchip RK3568 (RKNN)

本文档是 `algorithms/AGENTS.md`（主算法包规范）的硬件适配补充协议。在开发针对瑞芯微 RK3568 平台的算法包时，除需遵守主规范的 ABI 契约和生命周期外，必须严格遵循以下硬件加速与零拷贝原则。

## 1. 核心加速目标：NPU 与 RGA

RK3568 平台包含 1T 算力的 NPU 和 2D 图像加速引擎 RGA。算法包**必须**尽最大可能将计算卸载到这些专用硬件，严禁在 CPU (ARM Cortex-A55) 上进行大规模的像素遍历或模型推理。

## 2. 基于 DMA 的零拷贝 (DMA Zero-Copy)

主平台 Engine 在处理摄像头流或 MPP 硬件解码后，会通过 `hw_buffer_desc_t` 结构体将底层图像的 `dma_fd`（DMA 文件描述符）传递给算法包的 `detector_infer` 接口。

* **禁止 CPU 内存映射处理**：**严禁**通过 `mmap` 将 `dma_fd` 映射到 CPU 虚拟内存，再转换为 `cv::Mat` 给 OpenCV 处理。在低功耗 ARM 架构下，这会导致致命的内存带宽瓶颈和高延迟。
* **直接对接 NPU/RGA**：算法必须通过文件描述符 `fd` 直接进行硬件级的数据传递。

## 3. RGA 硬件前处理加速 (Hardware Pre-processing)

模型推理通常需要对输入视频帧进行 Resize、Crop、颜色格式转换（如 NV12 转 RGB888 或 BGR888）以及 Normalization。

* **必须使用 librga**：在 RK3568 上，上述前处理操作**必须**使用 Rockchip 提供的 2D 图形加速库 `librga.so`。
* **流转方式**：
    1. 提取 `hw_buffer_desc_t` 中的源 `dma_fd`，通过 `rga_buffer_t` 包装为源数据。
    2. 申请一块用于模型输入的连续 DMA 内存（目的 buffer）。
    3. 调用 `im2d_resize`、`im2d_crop` 或 `imcvtcolor` 等 API 完成转换，数据直接在内存控制器与 RGA 之间流转，绕过 CPU。

## 4. RKNN Runtime 推理

* **模型格式**：模型必须在 PC 端通过 `RKNN-Toolkit2` 预先转换为 `.rknn` 格式，并建议完成 INT8 量化或 FP16 精度对齐。
* **零拷贝输入设置**：通过 `rknn_tensor_mem` 分配 NPU 输入内存，并在调用 `rknn_inputs_set` 时绑定该内存块（或复用 RGA 处理后的目的 DMA buffer），以实现从解码/ISP到NPU推理的端到端零拷贝。
* **上下文生命周期**：
  * 在 `detector_init` 中调用 `rknn_init` 加载模型并分配专用的内存。
  * 在 `detector_destroy` 中必须严格调用 `rknn_destroy` 和内存释放 API，防止连续重启算法导致系统 CMA 内存枯竭。

## 5. 交叉编译与系统依赖 (Cross-Compilation)

* **工具链**：必须使用适配 Rockchip 的 `aarch64-linux-gnu` GCC/G++ 交叉编译工具链。
* **CMake 依赖**：`CMakeLists.txt` 中必须引入对 `librknnrt.so` 和 `librga.so` 的查找和链接。
* **安全与异常处理**：在调用 RGA 和 RKNN API 时，必须检查其返回值。如果硬件调用失败，**必须**捕获异常并返回错误码，严禁任由进程崩溃（段错误会导致主 Engine 瘫痪）。

## 6. 元数据声明补充

在 `algo_meta.yaml` 中，除标准字段外：

* `capabilities`: 必须显式声明包含 `npu` (RKNN) 和 `rga`，以便主平台确认该算法包的硬件依赖。
