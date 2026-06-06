// ABI 接口测试
// 验证算法包的接口是否符合规范

#include <iostream>
#include <cassert>
#include <cstring>
#include <dlfcn.h>

// ABI 类型定义
typedef void* algo_handle_t;

struct hw_buffer_desc_t {
    int dma_fd;
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int dma_buf_fd;
    uint64_t phys_addr;
    void* user_data;
};

struct infer_result_t {
    char* result_json;
    size_t result_json_len;
    uint32_t infer_time_us;
    int reserved[4];
};

// 函数指针类型
typedef algo_handle_t (*detector_init_func)(const char* config_json);
typedef int (*detector_infer_func)(algo_handle_t handle,
                                  const hw_buffer_desc_t* input,
                                  const char* context_json,
                                  infer_result_t* result);
typedef void (*detector_free_result_func)(infer_result_t* result);
typedef void (*detector_destroy_func)(algo_handle_t handle);
typedef const char* (*detector_version_func)(void);
typedef const char* (*detector_name_func)(void);
typedef int (*detector_self_test_func)(void);

// PRD 接口
typedef void* (*create_detector_func)(const char* package_dir);
typedef char* (*detector_infer_v2_func)(void* detector, 
                                       const void* image_array,
                                       const char* ai_params_json);
typedef void (*detector_free_result_v2_func)(char* result);
typedef void (*destroy_detector_func)(void* detector);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <nikoniko_detector.so 路径>" << std::endl;
        return 1;
    }
    
    const char* so_path = argv[1];
    std::cout << "=== ABI 接口测试 ===" << std::endl;
    std::cout << "加载: " << so_path << std::endl;
    
    // 加载动态库
    void* handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "❌ 无法加载动态库: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "✅ 动态库加载成功" << std::endl;
    
    // 检查必需符号
    auto init_fn = (detector_init_func)dlsym(handle, "detector_init");
    auto infer_fn = (detector_infer_func)dlsym(handle, "detector_infer");
    auto free_fn = (detector_free_result_func)dlsym(handle, "detector_free_result");
    auto destroy_fn = (detector_destroy_func)dlsym(handle, "detector_destroy");
    
    if (!init_fn) {
        std::cerr << "❌ 缺少符号: detector_init" << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✅ detector_init 存在" << std::endl;
    
    if (!infer_fn) {
        std::cerr << "❌ 缺少符号: detector_infer" << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✅ detector_infer 存在" << std::endl;
    
    if (!free_fn) {
        std::cerr << "❌ 缺少符号: detector_free_result" << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✅ detector_free_result 存在" << std::endl;
    
    if (!destroy_fn) {
        std::cerr << "❌ 缺少符号: detector_destroy" << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✅ detector_destroy 存在" << std::endl;
    
    // 检查可选符号
    auto version_fn = (detector_version_func)dlsym(handle, "detector_version");
    auto name_fn = (detector_name_func)dlsym(handle, "detector_name");
    auto self_test_fn = (detector_self_test_func)dlsym(handle, "detector_self_test");
    
    if (version_fn) {
        std::cout << "✅ detector_version: " << version_fn() << std::endl;
    }
    if (name_fn) {
        std::cout << "✅ detector_name: " << name_fn() << std::endl;
    }
    if (self_test_fn) {
        std::cout << "✅ detector_self_test 存在" << std::endl;
    }
    
    // 检查 PRD 接口（向后兼容）
    auto create_fn = (create_detector_func)dlsym(handle, "create_detector");
    auto infer_v2_fn = (detector_infer_v2_func)dlsym(handle, "detector_infer_v2");
    auto free_v2_fn = (detector_free_result_v2_func)dlsym(handle, "detector_free_result_v2");
    auto destroy_v2_fn = (destroy_detector_func)dlsym(handle, "destroy_detector");
    
    if (create_fn) {
        std::cout << "✅ create_detector (PRD接口) 存在" << std::endl;
    }
    if (infer_v2_fn) {
        std::cout << "✅ detector_infer_v2 (PRD接口) 存在" << std::endl;
    }
    if (free_v2_fn) {
        std::cout << "✅ detector_free_result_v2 (PRD接口) 存在" << std::endl;
    }
    if (destroy_v2_fn) {
        std::cout << "✅ destroy_detector (PRD接口) 存在" << std::endl;
    }
    
    // 测试参数验证
    std::cout << "\n=== 参数验证测试 ===" << std::endl;
    
    // 测试空指针
    int ret = init_fn(nullptr);
    if (ret == 0) {
        std::cerr << "❌ detector_init(nullptr) 应该返回 nullptr" << std::endl;
    } else {
        std::cout << "✅ detector_init(nullptr) 正确返回 nullptr" << std::endl;
    }
    
    // 测试无效 JSON
    ret = (int)(intptr_t)init_fn("invalid json");
    if (ret != 0) {
        std::cerr << "❌ detector_init(invalid json) 应该返回 nullptr" << std::endl;
    } else {
        std::cout << "✅ detector_init(invalid json) 正确返回 nullptr" << std::endl;
    }
    
    // 测试空 package_dir
    ret = (int)(intptr_t)init_fn(R"({"package_dir":""})");
    if (ret != 0) {
        std::cerr << "❌ detector_init(empty package_dir) 应该返回 nullptr" << std::endl;
    } else {
        std::cout << "✅ detector_init(empty package_dir) 正确返回 nullptr" << std::endl;
    }
    
    // 测试 detector_infer 参数验证
    infer_result_t result;
    ret = infer_fn(nullptr, nullptr, nullptr, nullptr);
    if (ret == 0) {
        std::cerr << "❌ detector_infer(nullptr...) 应该返回错误" << std::endl;
    } else {
        std::cout << "✅ detector_infer(nullptr...) 正确返回错误码: " << ret << std::endl;
    }
    
    // 测试 detector_free_result
    result.result_json = strdup("test");
    result.result_json_len = 4;
    free_fn(&result);
    if (result.result_json != nullptr) {
        std::cerr << "❌ detector_free_result 应该将 result_json 设为 nullptr" << std::endl;
    } else {
        std::cout << "✅ detector_free_result 正确释放内存" << std::endl;
    }
    
    // 清理
    dlclose(handle);
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}
