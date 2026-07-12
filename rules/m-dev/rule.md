# Apple Silicon (Mac M 系列) Dev Rule

在开发针对 Apple M 系列芯片（M1/M2/M3/M4）的算法包时，除需遵守主规范的 ABI 契约和生命周期外，必须严格遵循以下硬件加速原则。

## 1. 核心加速目标：UMA 与 神经网络引擎 (ANE)

Apple Silicon 采用统一内存架构 (Unified Memory Architecture, UMA)，CPU、GPU 和 Apple Neural Engine (ANE) 共享同一块物理内存。算法包必须利用这一特性实现真正的零拷贝。

## 2. 统一内存与零拷贝 (Zero-Copy on UMA)

主平台 Engine 传递的 `hw_buffer_desc_t` 通常底层封装的是 `CVPixelBuffer` 或 `IOSurface`（Mac 平台的标准图像缓冲）。

* **避免冗余拷贝**：严禁在传递给推理引擎之前，将数据进行多余的系统内存拷贝。必须直接利用指针或通过 Metal / CoreVideo 框架进行映射对接。
* **内存对齐与格式**：确保输入视频流的 `pixel_format` 满足 CoreML 或 Metal 的直接读取要求（如 `kCVPixelFormatType_32BGRA` 或 `kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange`）。

## 3. 推理后端的选择与加速

在 Mac 平台上，优先使用以下官方支持或原生优化的引擎加载模型：

1. **CoreML Native (最佳选择)**：
    * 通过 Objective-C++ (`.mm`) 直接加载 `.mlmodelc` 或 `.mlpackage`。
    * 此方式能最完美地自动调度 ANE (NPU) 和 GPU 算力。
2. **ONNXRuntime (使用 CoreML EP)**：
    * 如果模型只能维持 ONNX 格式，必须在 `detector_init` 中强制启用 CoreML Execution Provider (EP)。
    * 在创建 Session Options 时追加 `OrtSessionOptionsAppendExecutionProvider_CoreML` 标志。
3. **Metal Performance Shaders (MPS)**：
    * 若需编写自定义算子或图像前处理（如复杂的颜色转换、缩放），必须使用 Metal 编写 Compute Shader 并在 GPU 上异步执行，严禁使用纯 CPU (`cv::Mat`) 处理高分辨率视频流。

## 4. CMake 构建与框架链接

* **语言混编**：由于需要对接 Apple 底层 API，部分源码需使用 Objective-C++ (`.mm`)，并使用 `extern "C"` 导出主规范要求的 ABI 接口。
* **依赖声明**：`CMakeLists.txt` 中必须显式链接 Apple 原生框架：

    ```cmake
    find_library(COREML_FRAMEWORK CoreML)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(your_algo PRIVATE ${COREML_FRAMEWORK} ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    ```

* **C++ 异常与 Objective-C 异常隔离**：在 C-ABI 的最外层（如 `detector_infer`），不仅需要 `catch (std::exception&)`，还需要捕获 `@catch (NSException *e)`，以防 Apple 框架的异常穿透导致 Go 宿主 Panic。

## 5. 元数据声明补充

在 `algo_meta.yaml` 中，除标准字段外：

* `capabilities`: 必须显式声明对 `coreml`, `metal` 或 `ane` 的支持级别，以便平台识别其硬件依赖属性。
