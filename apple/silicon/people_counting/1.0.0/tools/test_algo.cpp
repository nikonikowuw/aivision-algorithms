/**
 * @file test_algo.cpp
 * @brief Dynamic loader test tool for people_counting C ABI
 */

#include <iostream>
#include <dlfcn.h>
#include <string>
#include "engine/include/algo/abi_contract.h"

typedef algo_handle_t (*InitFunc)(const char*);
typedef int (*InferFunc)(algo_handle_t, const hw_buffer_desc_t*, const char*, infer_result_t*);
typedef void (*DestroyFunc)(algo_handle_t);
typedef const char* (*VersionFunc)(void);
typedef const char* (*NameFunc)(void);
typedef int (*SelfTestFunc)(void);
typedef void (*FreeResultFunc)(infer_result_t*);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_people_counting.so>" << std::endl;
        return 1;
    }

    std::string lib_path = argv[1];
    std::cout << "Loading library: " << lib_path << std::endl;

    void* handle = dlopen(lib_path.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << "Failed to dlopen library: " << dlerror() << std::endl;
        return 1;
    }

    auto init_fn = reinterpret_cast<InitFunc>(dlsym(handle, "detector_init"));
    auto infer_fn = reinterpret_cast<InferFunc>(dlsym(handle, "detector_infer"));
    auto destroy_fn = reinterpret_cast<DestroyFunc>(dlsym(handle, "detector_destroy"));
    auto version_fn = reinterpret_cast<VersionFunc>(dlsym(handle, "detector_version"));
    auto name_fn = reinterpret_cast<NameFunc>(dlsym(handle, "detector_name"));
    auto self_test_fn = reinterpret_cast<SelfTestFunc>(dlsym(handle, "detector_self_test"));
    auto free_fn = reinterpret_cast<FreeResultFunc>(dlsym(handle, "detector_free_result"));

    if (!init_fn || !infer_fn || !destroy_fn || !version_fn || !name_fn || !self_test_fn) {
        std::cerr << "Failed to locate one or more required symbols." << std::endl;
        dlclose(handle);
        return 1;
    }

    std::cout << "✓ Located all symbols successfully." << std::endl;
    std::cout << "Algorithm Name:    " << name_fn() << std::endl;
    std::cout << "Algorithm Version: " << version_fn() << std::endl;

    std::cout << "Running detector_self_test..." << std::endl;
    int test_res = self_test_fn();
    if (test_res != 0) {
        std::cerr << "✗ detector_self_test failed with code: " << test_res << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << "✓ detector_self_test passed." << std::endl;

    dlclose(handle);
    std::cout << "Test completed successfully." << std::endl;
    return 0;
}
