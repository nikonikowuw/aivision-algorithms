#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>

using algo_handle_t = void*;

struct hw_buffer_desc_t {
    int dma_fd;
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int dma_buf_fd;
    uint64_t phys_addr;
};

struct infer_result_t {
    char* result_json;
    size_t result_json_len;
    uint32_t infer_time_us;
    int reserved[4];
};

using detector_init_func = algo_handle_t (*)(const char* config_json);
using detector_infer_func = int (*)(algo_handle_t handle, const hw_buffer_desc_t* input, const char* context_json, infer_result_t* result);
using detector_free_result_func = void (*)(infer_result_t* result);
using detector_destroy_func = void (*)(algo_handle_t handle);
using detector_self_test_func = int (*)(void);
using detector_version_func = const char* (*)(void);
using detector_name_func = const char* (*)(void);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <nikoniko_detector.so>" << std::endl;
        return 1;
    }

    void* so = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::cerr << "✗ dlopen 失败: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✓ 动态库加载成功" << std::endl;

    auto init_fn = reinterpret_cast<detector_init_func>(dlsym(so, "detector_init"));
    auto infer_fn = reinterpret_cast<detector_infer_func>(dlsym(so, "detector_infer"));
    auto free_fn = reinterpret_cast<detector_free_result_func>(dlsym(so, "detector_free_result"));
    auto destroy_fn = reinterpret_cast<detector_destroy_func>(dlsym(so, "detector_destroy"));
    auto self_test_fn = reinterpret_cast<detector_self_test_func>(dlsym(so, "detector_self_test"));
    auto version_fn = reinterpret_cast<detector_version_func>(dlsym(so, "detector_version"));
    auto name_fn = reinterpret_cast<detector_name_func>(dlsym(so, "detector_name"));

    if (!init_fn || !infer_fn || !free_fn || !destroy_fn || !self_test_fn || !version_fn || !name_fn) {
        std::cerr << "✗ ABI 符号不完整" << std::endl;
        dlclose(so);
        return 1;
    }
    std::cout << "✓ ABI 符号完整" << std::endl;
    std::cout << "✓ name=" << name_fn() << ", version=" << version_fn() << std::endl;

    if (init_fn(nullptr) != nullptr) {
        std::cerr << "✗ detector_init(nullptr) 应返回 nullptr" << std::endl;
        dlclose(so);
        return 1;
    }
    std::cout << "✓ 空配置校验通过" << std::endl;

    algo_handle_t handle = init_fn("{}");
    if (!handle) {
        std::cerr << "✗ detector_init({}) 应从动态库目录推导 package_dir 并初始化成功" << std::endl;
        dlclose(so);
        return 1;
    }
    destroy_fn(handle);
    std::cout << "✓ 默认 package_dir 初始化通过" << std::endl;

    infer_result_t result{};
    if (infer_fn(nullptr, nullptr, nullptr, &result) == 0) {
        std::cerr << "✗ detector_infer 空 handle 应返回错误" << std::endl;
        dlclose(so);
        return 1;
    }
    std::cout << "✓ infer 参数校验通过" << std::endl;

    result.result_json = static_cast<char*>(std::malloc(3));
    std::strcpy(result.result_json, "[]");
    result.result_json_len = 2;
    free_fn(&result);
    if (result.result_json != nullptr || result.result_json_len != 0) {
        std::cerr << "✗ detector_free_result 未清空结果" << std::endl;
        dlclose(so);
        return 1;
    }
    std::cout << "✓ result 内存释放通过" << std::endl;

    dlclose(so);
    std::cout << "=== ABI 测试通过 ===" << std::endl;
    return 0;
}
