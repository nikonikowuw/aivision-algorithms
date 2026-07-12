# Oracle

本文件定义智能检测系统 C++ 算法包的推荐目录架构、设计原则与编译构建规范。算法开发应严格遵循物理分层，以确保代码具有良好的可维护性、高内聚低耦合以及跨平台可移植性。

---

## 1. 推荐目录架构

算法包根目录（例如 `face/`、`yolov8/` 等）应按如下结构进行组织，严禁将所有源文件无序平铺在根目录下：

```text
algorithms/<algorithm_name>/<version>/         # 算法包根目录（如 face/）
├── CMakeLists.txt         # 根构建文件
├── Makefile                # 快捷编译脚本
├── algo_meta.yaml         # 必需：C++ Engine 元数据
├── label_map.json         # 必需：类别映射文件
├── testimage.jpg          # 必需：自检图片
├── include/               # 存放对外导出的公共头文件
│   └── algo_api.h         # （可选）对外导出的 API 声明
├── weights/               # 模型权重目录（如 .om 或 .rknn 文件）
├── tools/                 # 自检与测试工具（如 test_algo.cpp, bench_algo.cpp）
├── third_party/           # 第三方依赖库（不易通过系统安装的静态库/头文件）
├── .env.example           # 环境变量， 配置算法中参数的默认值
└── src/                   # 源代码根目录
    ├── CMakeLists.txt     # 源码层构建文件
    ├── common/            # 基础公共模块（日志、配置解析、计时器、错误处理、JSON 辅助等）
    ├── runtime/           # 硬件运行期管理（平台初始化、内存管理、通用模型加载与前向控制）
    ├── preprocess/        # 预处理模块（Resize、LetterBox、色彩空间转换、AIPP 配置等）
    ├── models/            # 核心模型封装（YOLO、SCRFD、AdaFace 等推理输入输出对齐）
    ├── postprocess/       # 后处理模块（NMS、YOLO Decode、分类/特征解析、人脸质量计算等）
    ├── pipeline/          # 业务流水线层（级联检测、目标追踪、特征融合等核心业务逻辑）
    └── app/               # C ABI 导出入口（algo_api.cpp，定义 algo_init/infer/destroy）
```

---

## 2. 核心架构设计原则

### 2.1 单向物理依赖 (Strict Unidirectional Dependency)

- 依赖关系应严格自上而下：
  $$\text{app (C ABI)} \rightarrow \text{pipeline} \rightarrow \text{[models, preprocess, postprocess]} \rightarrow \text{runtime} \rightarrow \text{common}$$
- 严禁出现循环依赖（例如 `models` 反向调用 `pipeline` 中的业务结构，或者 `common` 依赖具体的模型实现）。

### 2.2 职责单一与硬件细节隔离

- **屏蔽硬件细节**：`pipeline` 业务逻辑不应直接操作 `aclrtStream` 或 `rknn_context` 等平台原生句柄，这些句柄应由 `runtime` 封装。
- **预处理/后处理解耦**：模型推理前后对数据的处理（如 Resize、NMS）必须封装在 `preprocess` 与 `postprocess` 模块，`models` 仅负责加载模型和执行 Forward 推理。

### 2.3 平台抽象与可移植性

- 系统支持多款边缘硬件（如 Huawei Ascend 和 Rockchip）。
- 尽量将通用的数学计算、追踪算法、特征比对逻辑放入平台无关的公共模块；将涉及特定平台硬件加速器（如昇腾 DVPP、瑞芯微 RGA）的部分局限在 `preprocess` 或 `runtime` 中，方便在不同平台间复用业务 pipeline。

---

## 3. 模块化构建规范（Target-based Modern CMake）

推荐采用模块化静态库再链接成最终动态库的构建方案，避免把所有源文件一次性堆入单个动态库。

### 3.1 根目录与源码目录构建

在 `src/CMakeLists.txt` 中，推荐按子目录将模块划分为静态库目标：

```cmake
# src/CMakeLists.txt
add_subdirectory(common)
add_subdirectory(runtime)
add_subdirectory(preprocess)
add_subdirectory(models)
add_subdirectory(postprocess)
add_subdirectory(pipeline)
add_subdirectory(app)
```

### 3.2 静态库目标定义示例

以 `src/common/CMakeLists.txt` 为例：

```cmake
# src/common/CMakeLists.txt
add_library(algo_common STATIC
    config_parser.cpp
    json_utils.cpp
)

target_include_directories(algo_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)
```

### 3.3 导出共享库目标定义示例

在 `src/app/CMakeLists.txt` 中，将各静态库聚合链接，并输出符合加载契约的动态库：

```cmake
# src/app/CMakeLists.txt
add_library(tentcoo_detection SHARED
    algo_api.cpp
)

# 显式链接依赖的局部静态库和平台库
target_link_libraries(tentcoo_detection PRIVATE
    algo_pipeline
    algo_models
    algo_postprocess
    algo_preprocess
    algo_runtime
    algo_common
    ${ASCEND_LIBS} # 或者是 ${RKNN_LIBS} 等硬件库
    dl
)

# 强制去除 lib 前缀，以满足 Engine 针对 tentcoo_detection.so 的加载契约
set_target_properties(tentcoo_detection PROPERTIES PREFIX "")

# 将编译好的动态库输出到根目录，便于打包
install(TARGETS tentcoo_detection DESTINATION ${PROJECT_SOURCE_DIR})
```

### 3.4 模块化 CMake 的收益

1. **物理边界清晰**：每个子模块只能访问其 `CMakeLists.txt` 中显式指定的依赖，强迫开发者在代码层面解耦。
2. **增量编译加速**：只修改某个后处理算法时，无需重新编译整个工程的源文件，仅需重编译 `algo_postprocess` 并重新链接动态库。
3. **单元测试便利**：可针对 `algo_postprocess` 等单独的静态库编写独立的测试可执行程序，无需初始化完整的推理硬件环境。
