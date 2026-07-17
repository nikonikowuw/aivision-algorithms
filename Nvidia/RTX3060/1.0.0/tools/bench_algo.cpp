/**
 * @file bench_algo.cpp
 * @brief GPU Face Recognition — Benchmark program.
 *        Loads nikoniko_detector.so and runs multiple inference cycles for timing.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <dlfcn.h>

// Mirror ABI types
typedef struct algo_context_t* algo_handle_t;
typedef struct { void* data; uint32_t width; uint32_t height; uint32_t pixel_format; uint32_t stride; } bench_hw_buffer;
typedef struct { char* result_json; size_t result_json_len; uint32_t infer_time_us; int reserved[4]; } bench_result;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_nikoniko_detector.so> [num_iterations]\n", argv[0]);
        return 1;
    }

    const char* lib_path = argv[1];
    int num_iterations = (argc > 2) ? atoi(argv[2]) : 100;

    printf("=== GPU Face Recognition Benchmark ===\n");
    printf("Library: %s\n", lib_path);
    printf("Iterations: %d\n\n", num_iterations);

    // Load library
    void* handle = dlopen(lib_path, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    // Load function pointers
    auto init_fn = reinterpret_cast<algo_handle_t(*)(const char*)>(dlsym(handle, "detector_init"));
    auto infer_fn = reinterpret_cast<int(*)(algo_handle_t, const bench_hw_buffer*, const char*, bench_result*)>(dlsym(handle, "detector_infer"));
    auto destroy_fn = reinterpret_cast<void(*)(algo_handle_t)>(dlsym(handle, "detector_destroy"));
    auto free_fn = reinterpret_cast<void(*)(bench_result*)>(dlsym(handle, "algo_free_result"));

    if (!init_fn || !infer_fn || !destroy_fn || !free_fn) {
        fprintf(stderr, "Failed to load required symbols\n");
        dlclose(handle);
        return 1;
    }

    // Initialize
    printf("Initializing algorithm...\n");
    const char* config = "{\"package_dir\": \".\"}";
    algo_handle_t algo = init_fn(config);
    if (!algo) {
        fprintf(stderr, "Failed to initialize algorithm\n");
        dlclose(handle);
        return 1;
    }
    printf("✓ Algorithm initialized\n\n");

    // Create dummy input (640×480 BGR)
    bench_hw_buffer input{};
    input.width = 640;
    input.height = 480;
    input.stride = 640;
    input.data = std::calloc(640 * 480 * 3, 1);  // Black frame

    // Benchmark
    printf("Running %d iterations...\n", num_iterations);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_iterations; ++i) {
        bench_result result{};
        infer_fn(algo, &input, nullptr, &result);
        free_fn(&result);
    }

    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / num_iterations;
    double fps = 1000.0 / avg_ms;

    printf("\n=== Benchmark Results ===\n");
    printf("Total time: %.2f ms\n", total_ms);
    printf("Average per frame: %.2f ms\n", avg_ms);
    printf("Throughput: %.1f FPS\n", fps);

    // Cleanup
    std::free(input.data);
    destroy_fn(algo);
    dlclose(handle);

    printf("\n✓ Benchmark completed\n");
    return 0;
}
