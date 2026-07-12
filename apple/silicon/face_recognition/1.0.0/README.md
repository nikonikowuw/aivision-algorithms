# M1 Pro Face Recognition Algorithm Package (OpenCV-Free)

This is a highly optimized, multi-stage face recognition pipeline package designed for Apple Silicon (M1 Pro). It is built entirely in **pure C++ without OpenCV dependencies** to ensure a minimal memory footprint, fast load times, and self-contained execution on edge nodes.

---

## 🚀 Key Features & Optimizations

* **Zero OpenCV Dependency**:
  All image manipulation, color space conversion, resizing, and warping are implemented in optimized C++:
  * **NV12 to BGR**: Directly decodes raw YUV420sp (NV12) stream buffers to BGR bytes using BT.601 color space coefficients.
  * **Bilinear Resize**: Fast interpolation supporting custom pitches and pitch-linear strides.
  * **Letterbox**: Preserves the original aspect ratio by scaling and placing the image inside standard square dimensions, padding boundary margins with gray (`114`).
  * **Single-Pass Planar Blobbing**: Combines BGR-to-RGB channel swapping, floating-point mapping, and mean/std normalization `(x - 127.5) / 127.5` directly while formatting CHW planar layouts.
* **Math & Geometry Solvers**:
  * **Closed-Form Similarity Solver**: Resolves the optimal similarity transformation parameters (scale, rotation, translation) for 5-point face landmarks mapping onto standard reference templates using least squares.
  * **Bilinear Warp Affine**: Inverse maps target coordinate outputs back to source pixel grids, interpolating BGR values with sub-pixel accuracy.
  * **Greedy NMS**: Implements standard greedy Non-Maximum Suppression (NMS) matching `cv::dnn::NMSBoxes` behavior.
* **M1 Pro Hardware Acceleration**:
  * Employs **ONNX Runtime** integrated with **XNNPACK** and native **Apple CoreML** execution backends (`.mlpackage` models).
  * Wraps pitch-linear BGR input and head ROIs as non-owning image views. NV12 input uses a stride-aware CPU conversion before inference.
  * Reuses model input, resize, alignment, and output buffers across frames to avoid steady-state heap churn.

---

## 📐 Pipeline Architecture

```text
Input Video Frame (NV12 / BGR)
       │
       ▼
   NV12ToBGR (if NV12) ──▶ Wrap BGR Image view
       │
       ▼
 1. Body Detection (SCRFD Person, 640x640)
       │
       ▼
 2. ByteTracker (Association of body track IDs)
       │
       ▼
 3. Adaptive Head ROI Cropping (Dynamic crop based on body bounding box)
       │
       ▼
 4. Face Detection (SCRFD 2.5G, 160x160)
       │
       ▼
 5. Face Alignment (Closed-form Similarity Solver & Warp Affine to 112x112)
       │
       ▼
 6. Feature Extraction (AdaFace 512-d embeddings)
       │
       ▼
 7. Face Vector Index (Cosine Similarity search on snapshot database)
       │
       ▼
 Output JSON Results
```

---

## 📁 Directory Structure

```text
.
├── CMakeLists.txt         # CMake build configuration
├── Makefile               # Make workflow encapsulation
├── README.md              # Documentation
├── algo_meta.yaml         # Algorithm package metadata (params, result schemas)
├── label_map.json         # Label mapping configuration
├── testimage.jpg          # Validation image
├── .env                   # Environment configurations
├── src/                   # C++ Source code
│   ├── algo/              # ABI entry points and standard interface contract
│   ├── app/               # Dynamic library entry code (.mm / .cpp)
│   ├── common/            # Logging, config parser, and core types
│   ├── models/            # SCRFD and AdaFace model wrappers
│   ├── pipeline/          # ByteTracker and main orchestrator
│   ├── postprocess/       # Custom NMS, Face Align, and coordinate mapping
│   ├── preprocess/        # Custom NV12, Resizer, and Letterbox utils
│   └── runtime/           # ONNX Runtime & CoreML wrappers
├── weights/               # Native CoreML weights (.mlpackage format)
├── models/                # ONNX weights (used for ABI self-testing and packaging)
└── tests/                 # ABI validation and self-test harness
```

---

## 💼 C ABI Interface Contract

The shared library `nikoniko_detector.so` exports standard C-ABI functions to easily plug into the Go control plane:

* `detector_init(const char* config_json)`: Instantiates and initializes the orchestrator and runtime contexts.
* `detector_infer(algo_handle_t handle, const hw_buffer_desc_t* input, const char* context_json, infer_result_t* result)`: Runs full video inference on raw image memory and returns JSON results.
* `detector_update_face_library(algo_handle_t handle, const char* face_library_json)`: Updates the in-memory face database snapshot.
* `algo_free_result(infer_result_t* result)`: Safely frees memory allocated for inference output strings.
* `detector_destroy(algo_handle_t handle)`: Releases all models, runtimes, and system threads.
* `detector_self_test()`: Runs verification pipelines on local sample resources.
* `detector_version()`: Returns package version metadata (`1.0.0`).
* `detector_name()`: Returns algorithm namespace (`face_recognition_m1_pro`).

### Future Engine ABI: Apple Native Buffers

The current `hw_buffer_desc_t` contract provides a host `data` pointer and one row stride. It cannot fully describe a retained `CVPixelBuffer`/`IOSurface`, independent NV12 plane strides, native-buffer ownership, or producer/consumer synchronization. Consequently, BGR24 can be wrapped without copying, while NV12 currently requires a stride-aware CPU conversion.

A future Engine ABI revision should use the platform-reserved payload, with explicit version and size fields, to carry a borrowed native handle, buffer kind, FourCC, plane count, per-plane strides/offsets, and synchronization metadata. The intended path is:

```text
Decoder CVPixelBuffer -> Engine descriptor -> CoreML image input / CVMetalTextureCache
                      -> fused resize + color conversion + normalization -> inference
```

Compatibility requirements:

- Keep the existing `detector_infer` signature and packed BGR24 path.
- Define retain/release ownership for the complete synchronous call.
- Validate ABI version, payload size, native FourCC, plane count, and strides before use.
- Do not restore the `zero_copy_input` capability until Instruments confirms that no full-frame CPU copy occurs and the native-buffer integration tests pass.

The executable contract and required test matrix are documented in [Apple Native Buffer ABI](docs/apple-native-buffer-abi.md).

---

## 🛠 Compilation & Build Commands

All build, verification, and distribution tasks are encapsulated in standard Makefile targets:

### 1. Build Compilation

Compiles the shared library `nikoniko_detector.so` with Release flags:

```bash
make build
```

### 2. Validation Checks

Performs file checks, validates exported symbol tables using `nm`, and parses metadata schemas via Python:

```bash
make validate
```

### 3. ABI Integration Testing

Compiles a client application testing ABI symbol loads, performs a self-test cycle on `testimage.jpg`, and runs the CTyped self-test program:

```bash
make test
```

### 4. Deploy Package Creation

Builds, validates, and archives all required weights, libraries, and schemas into a standardized distribution bundle:

```bash
make package
```

### 5. Cleaning

Removes temporary build directories and artifact files:

```bash
make clean
```
