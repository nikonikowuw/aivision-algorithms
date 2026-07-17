# Apple Silicon (Mac M 系列) People Counting Algorithm Package

This is a highly optimized people counting algorithm package designed for Apple Silicon.

## 🚀 Key Features & Optimizations
- **YOLOv8n + CoreML**: Uses Apple Silicon Neural Engine (ANE) for hardware accelerated object detection.
- **ByteTracker**: CPU-based, lightweight, multi-object tracking.
- **Line Crossing**: Supports dynamic crossing line configurations from `context_json` per frame.
- **Pure C++ / ObjC++**: Self-contained C ABI wrapper.

## 💼 C ABI Interface Contract
- `detector_init(const char* config_json)`
- `detector_infer(algo_handle_t handle, const hw_buffer_desc_t* input, const char* context_json, infer_result_t* result)`
- `detector_destroy(algo_handle_t handle)`
- `detector_version()`
- `detector_name()`
- `detector_self_test()`
