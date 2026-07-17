# Apple Silicon (Mac M 系列) Safety Helmet Detection Algorithm Package

This is a highly optimized safety helmet (helmet) detection algorithm package designed for Apple Silicon (macOS ARM64).

## 🚀 Key Features & Optimizations
- **DAMO-YOLO-S + CoreML EP**: Uses Apple Silicon Neural Engine (ANE) via ONNXRuntime's CoreML Execution Provider for hardware-accelerated object detection.
- **Zero-Copy on UMA**: Leverages Unified Memory Architecture (UMA) to eliminate redundant frame copies during buffer conversion.
- **Pure C++ / ObjC++**: Self-contained C ABI wrapper mapping to `tentcoo_detection.so` per the Engine loading contract.
- **Robust Exception Handling**: Protects the Gin/Go host process from C++ exceptions or Apple framework NSExceptions.

## 💼 C ABI Interface Contract
- `detector_init(const char* config_json)`
- `detector_infer(algo_handle_t handle, const hw_buffer_desc_t* input, const char* context_json, infer_result_t* result)`
- `detector_destroy(algo_handle_t handle)`
- `detector_version()`
- `detector_name()`
- `detector_self_test()`

## 🏷️ Category Mapping
| Class ID | Class Name | Category Code | Description |
|---|---|---|---|
| 0 | `safety_hat` | `10001` | Person wearing a safety helmet |
| 1 | `no_safety_hat` | `10002` | Person not wearing a safety helmet |

## ⚙️ Configurable Parameters (`algo_meta.yaml`)
- `conf_threshold` (number, default: `0.5`): Confidence threshold to filter out low-confidence bounding boxes.
- `iou_threshold` (number, default: `0.45`): Non-Maximum Suppression (NMS) Intersection-over-Union (IoU) threshold.

## 🛠️ Build & Package
```bash
# Build the package (produces tentcoo_detection.so)
make build

# Validate ABI symbols and configurations
make validate

# Package into safety_helmet_1.0.0.zip
make package

# Clean build artifacts
make clean
```
