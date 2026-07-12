/**
 * @file test_abi.cpp
 * @brief ABI 冒烟测试 — 验证 C 接口的 dlopen/dlsym 动态加载链路
 *        ABI smoke test — validates the dlopen/dlsym dynamic loading chain for the C interface
 *
 * 测试场景（按执行顺序）：
 * 1. dlopen 加载动态库，dlsym 获取所有 ABI 符号
 * 2. detector_init(nullptr) 应返回 nullptr
 * 3. detector_init("{}") 空初始化
 * 4. detector_update_face_library 使用空人脸库 JSON
 * 5. algo_free_result 释放空结果
 * 6. detector_destroy 销毁句柄
 * 7. dlclose 卸载库
 *
 * Test scenarios (in execution order):
 * 1. dlopen loads the shared library, dlsym fetches all ABI symbols
 * 2. detector_init(nullptr) should return nullptr
 * 3. detector_init("{}") with empty JSON
 * 4. detector_update_face_library with empty face library JSON
 * 5. algo_free_result frees a null result
 * 6. detector_destroy destroys the handle
 * 7. dlclose unloads the library
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include "algo/abi_contract.h"

// Define function pointers matching the ABI
// 定义与 ABI 匹配的函数指针类型
typedef algo_handle_t (*detector_init_func)(const char* config_json);
typedef int (*detector_infer_func)(algo_handle_t handle, const hw_buffer_desc_t* input, const char* context_json, infer_result_t* result);
typedef int (*detector_update_face_library_func)(algo_handle_t handle, const char* face_library_json);
typedef void (*algo_free_result_func)(infer_result_t* result);
typedef void (*detector_destroy_func)(algo_handle_t handle);
typedef int (*detector_self_test_func)(void);
typedef const char* (*detector_version_func)(void);
typedef const char* (*detector_name_func)(void);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <nikoniko_detector.so>" << std::endl;
        return 1;
    }

    // Step 1: dlopen 加载动态库 / Load shared library via dlopen
    void* so = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::cerr << "dlopen failed: " << dlerror() << std::endl;
        return 1;
    }

    // Step 2: dlsym 获取所有 ABI 符号 / Fetch all ABI symbols via dlsym
    auto init_fn = reinterpret_cast<detector_init_func>(dlsym(so, "detector_init"));
    auto infer_fn = reinterpret_cast<detector_infer_func>(dlsym(so, "detector_infer"));
    auto update_face_library_fn = reinterpret_cast<detector_update_face_library_func>(dlsym(so, "detector_update_face_library"));
    auto free_fn = reinterpret_cast<algo_free_result_func>(dlsym(so, "algo_free_result"));
    auto destroy_fn = reinterpret_cast<detector_destroy_func>(dlsym(so, "detector_destroy"));
    auto self_test_fn = reinterpret_cast<detector_self_test_func>(dlsym(so, "detector_self_test"));
    auto version_fn = reinterpret_cast<detector_version_func>(dlsym(so, "detector_version"));
    auto name_fn = reinterpret_cast<detector_name_func>(dlsym(so, "detector_name"));

    // 验证所有符号已正确加载 / Verify all symbols loaded correctly
    if (!init_fn || !infer_fn || !update_face_library_fn || !free_fn || !destroy_fn || !self_test_fn || !version_fn || !name_fn) {
        std::cerr << "Missing ABI symbols" << std::endl;
        dlclose(so);
        return 1;
    }
    std::cout << "Successfully loaded library symbols: name=" << name_fn() << ", version=" << version_fn() << std::endl;

    // Step 3: 测试 detctor_init(nullptr) 应返回 nullptr
    // Test: detector_init(nullptr) should return nullptr
    if (init_fn(nullptr) != nullptr) {
        std::cerr << "detector_init(nullptr) should return nullptr" << std::endl;
        dlclose(so);
        return 1;
    }

    // Step 4: 测试空 JSON 初始化 / Test: empty JSON initialization
    algo_handle_t handle = init_fn("{}");
    if (!handle) {
        std::cerr << "detector_init(\"{}\") initialization failed" << std::endl;
        dlclose(so);
        return 1;
    }

    // Step 5: 测试空参数推理应失败 / Test: inference with null parameters should fail
    infer_result_t result{};
    if (infer_fn(nullptr, nullptr, nullptr, &result) == 0) {
        std::cerr << "detector_infer with null handle should fail" << std::endl;
        destroy_fn(handle);
        dlclose(so);
        return 1;
    }

    // Step 6: 测试空人脸库更新 / Test: update with empty face library
    if (update_face_library_fn(handle, R"({"version":"abi_test_empty","items":[]})") != 0) {
        std::cerr << "detector_update_face_library with empty snapshot failed" << std::endl;
        destroy_fn(handle);
        dlclose(so);
        return 1;
    }

    // Step 7: 测试释放结果 / Test: free result
    result.result_json = static_cast<char*>(std::malloc(3));
    std::strcpy(result.result_json, "[]");
    result.result_json_len = 2;
    free_fn(&result);
    if (result.result_json != nullptr || result.result_json_len != 0) {
        std::cerr << "detector_free_result did not clear result" << std::endl;
        destroy_fn(handle);
        dlclose(so);
        return 1;
    }

    // Step 8: 销毁句柄并卸载库 / Destroy handle and unload library
    destroy_fn(handle);
    dlclose(so);
    std::cout << "ABI test passed" << std::endl;
    return 0;
}
