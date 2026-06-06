# 修复记录

## 修复日期
2026-06-06

## 修复的问题

### 🔴 严重问题 (Blocking Issues)

#### 1. 修复 dma_fd 内存访问问题

**问题**: 原代码将 `dma_fd`（文件描述符）当作内存指针使用，导致段错误或数据损坏。

```cpp
// ❌ 错误代码
memcpy(image.data, reinterpret_cast<void*>(input->dma_fd), 
       input->width * input->height * 3);
```

**修复**: 使用 `mmap` 正确映射 DMA 缓冲区。

```cpp
// ✅ 修复后
if (desc->dma_fd >= 0) {
    void* mapped = mmap(nullptr, desc->size, PROT_READ, MAP_SHARED, desc->dma_fd, 0);
    if (mapped == MAP_FAILED) {
        return cv::Mat();
    }
    image = cv::Mat(desc->height, desc->width, CV_8UC3, mapped).clone();
    munmap(mapped, desc->size);
} else if (desc->user_data) {
    image = cv::Mat(desc->height, desc->width, CV_8UC3, 
                   static_cast<uint8_t*>(desc->user_data)).clone();
}
```

**文件**: `src/face_recognition.cpp`

---

#### 2. 修复 ABI 接口不匹配

**问题**: 项目使用 `detector_init/infer/destroy`，PRD 定义 `create_detector/infer/destroy`，接口不兼容。

**修复**: 同时导出两套接口，保持向后兼容。

```cpp
extern "C" {
    // 项目当前接口 (abi_contract.h)
    algo_handle_t detector_init(const char* config_json);
    int detector_infer(algo_handle_t handle, ...);
    void detector_free_result(infer_result_t* result);
    void detector_destroy(algo_handle_t handle);
    
    // PRD 定义接口 (向后兼容)
    void* create_detector(const char* package_dir);
    char* detector_infer_v2(void* detector, ...);
    void detector_free_result_v2(char* result);
    void destroy_detector(void* detector);
}
```

**文件**: `src/face_recognition.cpp`

---

#### 3. 实现 detector_free_result 函数

**问题**: `strdup` 分配的内存没有对应的释放函数。

**修复**: 实现 `detector_free_result` 函数。

```cpp
void detector_free_result(infer_result_t* result) {
    if (result && result->result_json) {
        free(result->result_json);  // 匹配 strdup 的 malloc
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}
```

**文件**: `src/face_recognition.cpp`

---

### 🟡 重要问题 (Important Issues)

#### 4. 添加完整的输入验证

**问题**: 参数验证过于简单，没有详细的错误码。

**修复**: 添加详细的参数验证和错误码。

```cpp
// 错误码定义
enum class AlgoError : int {
    SUCCESS = 0,
    INVALID_HANDLE = -1,
    INVALID_INPUT = -2,
    INVALID_OUTPUT = -3,
    INVALID_SIZE = -4,
    SIZE_TOO_LARGE = -5,
    MEMORY_MAP_FAILED = -6,
    NO_INPUT_DATA = -7,
    INIT_FAILED = -8,
    INFER_FAILED = -9,
    SELF_TEST_FAILED = -10,
};

// 参数验证
if (!handle) return static_cast<int>(AlgoError::INVALID_HANDLE);
if (!input) return static_cast<int>(AlgoError::INVALID_INPUT);
if (!result) return static_cast<int>(AlgoError::INVALID_OUTPUT);
if (input->width == 0 || input->height == 0) return static_cast<int>(AlgoError::INVALID_SIZE);
if (input->width > 8192 || input->height > 8192) return static_cast<int>(AlgoError::SIZE_TOO_LARGE);
```

**文件**: `src/face_recognition.cpp`

---

#### 5. 将硬编码阈值改为可配置

**问题**: 人脸识别阈值 `0.6` 是硬编码的。

**修复**: 添加 `recognition_threshold` 配置参数。

```cpp
// 配置结构
struct AlgorithmConfig {
    // ...
    float recognition_threshold = 0.6f;  // 新增
};

// 初始化时解析
ctx->config.recognition_threshold = std::max(0.0f, std::min(1.0f, 
    config.get("recognition_threshold", 0.6f).asFloat()));

// 使用配置值
if (best_similarity >= ctx->config.recognition_threshold && best_index >= 0) {
    // ...
}
```

**文件**: 
- `src/face_recognition.cpp`
- `include/face_recognition.h`
- `algo_meta.yaml`

---

### 🟢 改进建议 (Suggestions)

#### 6. 添加日志系统

**问题**: 使用 `std::cerr` 输出错误，没有级别控制。

**修复**: 添加日志宏。

```cpp
#define ALGO_LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define ALGO_LOG_WARN(fmt, ...)  fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__)
#define ALGO_LOG_INFO(fmt, ...)  fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__)
#define ALGO_LOG_DEBUG(fmt, ...) // 生产环境禁用
```

**文件**: `src/face_recognition.cpp`

---

#### 7. 修复自检路径问题

**问题**: `detector_self_test` 使用相对路径 "." 无法定位算法包。

**修复**: 从环境变量获取算法包目录。

```cpp
int detector_self_test() {
    const char* pkg_dir_env = getenv("ALGO_PACKAGE_DIR");
    std::string package_dir = pkg_dir_env ? pkg_dir_env : ".";
    // ...
}
```

**文件**: `src/face_recognition.cpp`

---

#### 8. 添加配置验证

**问题**: 配置参数没有范围验证。

**修复**: 添加参数范围验证。

```cpp
ctx->config.conf_thres = std::max(0.0f, std::min(1.0f, 
    config.get("conf_thres", 0.5f).asFloat()));
ctx->config.iou_thres = std::max(0.0f, std::min(1.0f, 
    config.get("iou_thres", 0.45f).asFloat()));
// ...
```

**文件**: `src/face_recognition.cpp`

---

## 新增文件

### 测试文件

- `tests/test_abi.cpp` - ABI 接口测试
- `tests/Makefile` - 测试编译配置

### 验证脚本

- `verify_fixes.sh` - 修复验证脚本

---

## 验证方法

运行验证脚本：

```bash
./verify_fixes.sh
```

运行 ABI 测试：

```bash
cd tests
make run
```

---

## 兼容性说明

### 向后兼容

- 保留了项目当前使用的 `detector_init/infer/destroy` 接口
- 新增了 PRD 定义的 `create_detector/infer_v2/destroy_detector` 接口
- 两套接口可以同时使用

### 配置兼容

- 所有新增配置参数都有默认值
- 旧的配置 JSON 仍然有效
- 新参数 `recognition_threshold` 是可选的

---

## 性能优化

### JSON Writer 复用

```cpp
struct AlgorithmContext {
    Json::StreamWriterBuilder writer_builder;  // 复用
    // ...
};
```

### 内存管理

- 使用 `strdup` + `free` 配对
- 明确的内存所有权
- 避免内存泄漏

---

## 安全改进

### 输入验证

- 检查所有指针参数
- 验证图像尺寸范围
- 防止缓冲区溢出

### 错误处理

- 详细的错误码
- 完整的异常捕获
- 资源清理保证
