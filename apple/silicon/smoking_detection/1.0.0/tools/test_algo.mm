// Smoking Detection runtime test: self-test plus explicit detector_infer.

#include <algo/abi_contract.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "picojson.h"

using InitFn = algo_handle_t (*)(const char*);
using InferFn = int (*)(algo_handle_t, const hw_buffer_desc_t*, const char*,
                        infer_result_t*);
using DestroyFn = void (*)(algo_handle_t);
using SelfTestFn = int (*)(void);
using FreeFn = void (*)(infer_result_t*);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s /path/to/tentcoo_detection.so\n", argv[0]);
        return 2;
    }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    auto init = reinterpret_cast<InitFn>(dlsym(library, "detector_init"));
    auto infer = reinterpret_cast<InferFn>(dlsym(library, "detector_infer"));
    auto destroy = reinterpret_cast<DestroyFn>(dlsym(library, "detector_destroy"));
    auto self_test = reinterpret_cast<SelfTestFn>(dlsym(library, "detector_self_test"));
    auto free_result = reinterpret_cast<FreeFn>(dlsym(library, "algo_free_result"));
    if (!init || !infer || !destroy || !self_test || !free_result) {
        std::fprintf(stderr, "required ABI symbol is missing\n");
        dlclose(library);
        return 1;
    }
    if (self_test() != 0) {
        std::fprintf(stderr, "detector_self_test failed\n");
        dlclose(library);
        return 1;
    }

    algo_handle_t handle = init("{}");
    if (!handle) {
        std::fprintf(stderr, "detector_init failed\n");
        dlclose(library);
        return 1;
    }
    constexpr uint32_t width = 640;
    constexpr uint32_t height = 640;
    std::vector<unsigned char> bgr(width * height * 3u, 0);
    hw_buffer_desc_t input{};
    input.size = bgr.size();
    input.width = width;
    input.height = height;
    input.pixel_format = 0;
    input.data = bgr.data();
    input.stride = width * 3u;
    input.buffer_type = HW_BUFFER_TYPE_DEFAULT;
    input.buffer_owner = HW_BUFFER_OWNER_ENGINE;

    infer_result_t result{};
    if (infer(handle, &input, nullptr, &result) != 0 || !result.result_json) {
        std::fprintf(stderr, "explicit detector_infer failed\n");
        destroy(handle);
        dlclose(library);
        return 1;
    }
    picojson::value parsed;
    const std::string parse_error = picojson::parse(parsed, result.result_json);
    if (!parse_error.empty() || !parsed.is<picojson::array>() ||
        result.result_json_len != std::strlen(result.result_json) ||
        result.infer_time_us == 0) {
        std::fprintf(stderr, "detector_infer returned an invalid result contract\n");
        free_result(&result);
        destroy(handle);
        dlclose(library);
        return 1;
    }
    free_result(&result);
    destroy(handle);
    destroy(handle);
    if (infer(handle, &input, nullptr, &result) == 0) {
        std::fprintf(stderr, "infer-after-destroy unexpectedly succeeded\n");
        dlclose(library);
        return 1;
    }
    dlclose(library);
    return 0;
}
