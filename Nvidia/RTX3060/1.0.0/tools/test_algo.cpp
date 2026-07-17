/**
 * @file test_algo.cpp
 * @brief GPU Face Recognition — ABI test program.
 *        Loads nikoniko_detector.so via dlopen/dlsym and validates ABI symbols.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

// Mirror ABI types locally
typedef struct algo_context_t* algo_handle_t;

typedef struct {
    int dma_fd;
    size_t size;
    uint32_t width;
    uint32_t height;
    uint32_t pixel_format;
    int dma_buf_fd;
    uint64_t phys_addr;
    void* data;
    uint32_t stride;
    uint32_t buffer_type;
    uint32_t buffer_owner;
    uint32_t reserved_flags;
    union {
        uint8_t plat[80];
        uint64_t plat_u64[10];
    };
} hw_buffer_desc_t; // sizeof = 144

typedef struct {
    char* result_json;
    size_t result_json_len;
    uint32_t infer_time_us;
    int reserved[4];
} infer_result_t; // sizeof = 40

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_nikoniko_detector.so>\n", argv[0]);
        return 1;
    }

    const char* lib_path = argv[1];
    printf("Loading shared library: %s\n", lib_path);

    void* handle = dlopen(lib_path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }
    printf("✓ Library loaded successfully\n\n");

    // ---- Verify struct sizes ----
    printf("=== ABI Struct Size Verification ===\n");
    if (sizeof(hw_buffer_desc_t) != 144) {
        fprintf(stderr, "✗ hw_buffer_desc_t size %zu != 144\n", sizeof(hw_buffer_desc_t));
        dlclose(handle);
        return 1;
    }
    printf("✓ sizeof(hw_buffer_desc_t) = %zu (expected 144)\n", sizeof(hw_buffer_desc_t));

    if (sizeof(infer_result_t) != 40) {
        fprintf(stderr, "✗ infer_result_t size %zu != 40\n", sizeof(infer_result_t));
        dlclose(handle);
        return 1;
    }
    printf("✓ sizeof(infer_result_t) = %zu (expected 40)\n\n", sizeof(infer_result_t));

    // ---- Load symbols ----
    printf("=== Symbol Loading ===\n");
    const char* symbols[] = {
        "detector_init",
        "detector_infer",
        "detector_destroy",
        "detector_version",
        "detector_name",
        "detector_self_test",
        "detector_update_face_library",
        "algo_free_result"
    };
    int num_symbols = sizeof(symbols) / sizeof(symbols[0]);

    for (int i = 0; i < num_symbols; ++i) {
        void* sym = dlsym(handle, symbols[i]);
        if (!sym) {
            fprintf(stderr, "✗ Missing symbol: %s — %s\n", symbols[i], dlerror());
            dlclose(handle);
            return 1;
        }
        printf("✓ symbol: %s @ %p\n", symbols[i], sym);
    }

    // ---- Test version/name ----
    printf("\n=== Version/Name Test ===\n");
    typedef const char* (*version_fn)(void);
    typedef const char* (*name_fn)(void);

    version_fn ver_fn = reinterpret_cast<version_fn>(dlsym(handle, "detector_version"));
    name_fn name_fn_ptr = reinterpret_cast<name_fn>(dlsym(handle, "detector_name"));

    if (ver_fn) printf("✓ version: %s\n", ver_fn());
    if (name_fn_ptr) printf("✓ name: %s\n", name_fn_ptr());

    // ---- Test self_test ----
    printf("\n=== Self-Test ===\n");
    typedef int (*self_test_fn)(void);
    self_test_fn st_fn = reinterpret_cast<self_test_fn>(dlsym(handle, "detector_self_test"));
    if (st_fn) {
        int ret = st_fn();
        if (ret == 0) {
            printf("✓ detector_self_test passed\n");
        } else {
            fprintf(stderr, "✗ detector_self_test failed with code %d\n", ret);
        }
    }

    dlclose(handle);
    printf("\n=== ABI Test Completed ===\n");
    return 0;
}
