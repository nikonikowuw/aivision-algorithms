#include <iostream>
#include <dlfcn.h>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "algo/abi_contract.h"

// Define function pointer types matching the contract
typedef algo_handle_t (*detector_init_fn)(const char *config_json);
typedef int (*detector_infer_fn)(algo_handle_t handle, const hw_buffer_desc_t *input,
                                 const char *context_json, infer_result_t *result);
typedef void (*detector_destroy_fn)(algo_handle_t handle);
typedef const char *(*detector_version_fn)(void);
typedef const char *(*detector_name_fn)(void);
typedef int (*detector_self_test_fn)(void);
typedef void (*algo_free_result_fn)(infer_result_t *result);

// Colors for terminal output
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"

int main() {
    std::cout << "=== Running Safety Helmet ABI Interface Test ===" << std::endl;

    // 1. Load library
    const char* lib_path = "./tentcoo_detection.so";
    void* handle = dlopen(lib_path, RTLD_LAZY);
    if (!handle) {
        std::cerr << ANSI_COLOR_RED << "FAIL: Failed to dlopen " << lib_path << ": " << dlerror() << ANSI_COLOR_RESET << std::endl;
        return 1;
    }
    std::cout << ANSI_COLOR_GREEN << "PASS: Successfully dlopen-ed " << lib_path << ANSI_COLOR_RESET << std::endl;

    // 2. Resolve symbols
    auto init_fn = (detector_init_fn)dlsym(handle, "detector_init");
    auto infer_fn = (detector_infer_fn)dlsym(handle, "detector_infer");
    auto destroy_fn = (detector_destroy_fn)dlsym(handle, "detector_destroy");
    auto version_fn = (detector_version_fn)dlsym(handle, "detector_version");
    auto name_fn = (detector_name_fn)dlsym(handle, "detector_name");
    auto self_test_fn = (detector_self_test_fn)dlsym(handle, "detector_self_test");
    auto free_result_fn = (algo_free_result_fn)dlsym(handle, "algo_free_result");

    if (!init_fn || !infer_fn || !destroy_fn || !version_fn || !name_fn || !self_test_fn || !free_result_fn) {
        std::cerr << ANSI_COLOR_RED << "FAIL: One or more required symbols are missing" << ANSI_COLOR_RESET << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << ANSI_COLOR_GREEN << "PASS: Resolved all required ABI symbols" << ANSI_COLOR_RESET << std::endl;

    // 3. Print name and version
    const char* name = name_fn();
    const char* version = version_fn();
    std::cout << "Algorithm Name: " << (name ? name : "NULL") << std::endl;
    std::cout << "Algorithm Version: " << (version ? version : "NULL") << std::endl;

    if (strcmp(name, "safety_helmet") != 0) {
        std::cerr << ANSI_COLOR_RED << "FAIL: Name mismatch. Expected 'safety_helmet', got '" << name << "'" << ANSI_COLOR_RESET << std::endl;
        dlclose(handle);
        return 1;
    }
    if (strcmp(version, "1.0.0") != 0) {
        std::cerr << ANSI_COLOR_RED << "FAIL: Version mismatch. Expected '1.0.0', got '" << version << "'" << ANSI_COLOR_RESET << std::endl;
        dlclose(handle);
        return 1;
    }

    // 4. Run self test
    std::cout << "Calling detector_self_test()..." << std::endl;
    int test_ret = self_test_fn();
    if (test_ret != 0) {
        std::cerr << ANSI_COLOR_RED << "FAIL: detector_self_test returned non-zero code: " << test_ret << ANSI_COLOR_RESET << std::endl;
        dlclose(handle);
        return 1;
    }
    std::cout << ANSI_COLOR_GREEN << "PASS: detector_self_test completed successfully" << ANSI_COLOR_RESET << std::endl;

    dlclose(handle);
    std::cout << ANSI_COLOR_GREEN << "=== All ABI Tests PASSED ===" << ANSI_COLOR_RESET << std::endl;
    return 0;
}
