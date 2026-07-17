# GPU Face Recognition Algorithm Package (NVIDIA RTX 3060)

A multi-stage face recognition pipeline optimized for NVIDIA GPU (RTX 3060, 12GB). Uses TensorRT FP16 inference with three deep learning models to perform real-time face recognition across 16 concurrent RTSP streams at 1-2 FPS per stream.

---

## Key Features

* **NVIDIA GPU Acceleration**: TensorRT FP16 inference on RTX 3060 with dynamic batching (1-16 streams)
* **Three-Model Pipeline**:
  * **YOLO11n** (640×640): Person detection with letterbox preprocessing
  * **RetinaFace** (640×640): Face detection + 5 landmarks
  * **AdaFace IR-50** (112×112): 512-d normalized face embedding extraction
* **Multi-Stream Batching**: BatchCollector with adaptive 8ms timeout scheduling
* **ByteTrack Tracking**: IoU-based multi-object tracking with face feature ID switch correction
* **FAISS Gallery Matching**: Scalable face identity search from 100 to 10M+ registered faces
* **GPU Face Alignment**: 5-point affine transform via NVIDIA NPP library
* **Zero OpenCV Dependency**: All image manipulation implemented in pure C++17

---

## Pipeline Architecture

```
16 RTSP Streams → CPU Decode (1-2 FPS)
       │
       ▼
BatchCollector (BatchSlot lock-free pool, 8ms timeout)
       │
       ▼
Batch Worker (GPU Single Thread)
       │
       ├─▶ YOLO11n (Person Detection, 640×640)
       │     └─▶ CPU Decode + NMS
       │
       ├─▶ RetinaFace (Face Detection + Landmarks, 640×640)
       │     └─▶ CPU Decode + NMS
       │
       ├─▶ IoU Association (Person ↔ Face)
       │
       ├─▶ Face Alignment (5-point Affine, GPU NPP → 112×112)
       │
       └─▶ AdaFace (Feature Extraction, 512-d L2-normalized)
              │
              ▼
       Matching Thread (CPU FAISS)
              │
              ├─▶ Short-term Store (std::vector, brute-force, <1ms)
              │
              └─▶ Gallery Index (FAISS: IndexFlat / IVFFlat / IVFPQ)
                     │
                     ▼
              Output JSON Results
```

---

## TensorRT Deployment

### Model Conversion

```bash
# YOLO11n — via Ultralytics export
yolo export model=yolo11n.pt format=engine half=true device=0

# RetinaFace — ONNX → TensorRT
trtexec --onnx=retinaface.onnx --saveEngine=retinaface.engine --fp16 \
  --optShapes=input:8x3x640x640 --minShapes=input:1x3x640x640 \
  --maxShapes=input:16x3x640x640

# AdaFace IR-50 — PyTorch → ONNX → TensorRT
trtexec --onnx=adaface_ir50.onnx --saveEngine=adaface_ir50.engine --fp16 \
  --optShapes=input:8x3x112x112 --minShapes=input:1x3x112x112 \
  --maxShapes=input:16x3x112x112
```

### Dynamic Batch Parameters

All engines use optimization profiles: `min=(1) opt=(8) max=(16)`. The `TrtBackend` sets actual batch size per `Run()` call.

---

## Directory Structure

```
.
├── CMakeLists.txt         # Root CMake build
├── Makefile               # Build / validate / package targets
├── algo_meta.yaml         # Package metadata
├── label_map.json         # Category code mapping
├── testimage.jpg          # Self-test image
├── README.md              # This file
├── .env.example           # Environment config template
├── include/
│   └── algo_api.h         # Optional public C API header
├── src/                   # C++ source code
│   ├── common/            # Types, config, logger, geometry utils
│   ├── runtime/           # TensorRT backend, CUDA utilities
│   ├── preprocess/        # Image resize, normalize, letterbox
│   ├── models/            # YOLO11n, RetinaFace, AdaFace wrappers
│   ├── postprocess/       # YOLO/Retina decoders, face aligner, coords
│   ├── pipeline/          # Orchestrator, tracker, batch collector, FAISS
│   └── app/               # C ABI exports (.so entry)
├── weights/               # TensorRT engine files + ONNX source
├── third_party/           # Third-party dependencies (FAISS headers)
└── tools/                 # test_algo, bench_algo utilities
```

---

## C ABI Interface Contract

The shared library `nikoniko_detector.so` exports standard C-ABI functions:

* `detector_init(config_json)`: Initialize algorithm context and load TensorRT engines
* `detector_infer(handle, input, context_json, result)`: Run full pipeline on one frame
* `detector_destroy(handle)`: Release all GPU/CPU resources
* `detector_version()`: Return version string (`"1.0.0"`)
* `detector_name()`: Return algorithm name (`"face_recognition_gpu"`)
* `detector_self_test()`: Verify ABI structures and run test image pipeline
* `detector_update_face_library(handle, json)`: Hot-update registered face gallery
* `algo_free_result(result)`: Free dynamically allocated result JSON

---

## Build & Package Commands

```bash
# Build shared library
make build

# Validate (file checks, symbol checks, metadata)
make validate

# Run ABI + self-test
make test

# Create distribution zip
make package

# Clean build artifacts
make clean
```

---

## FAISS Index Strategy

| Gallery Size | Index Type      | Search Latency |
|:-------------|:----------------|:---------------|
| 100 ~ 1K     | IndexFlat       | < 0.1 ms       |
| 1K ~ 100K    | IVFFlat         | ~ 1 ms         |
| 100K ~ 10M   | IVFPQ (M=64)    | ~ 5-20 ms      |

---

## Requirements

* **GPU**: NVIDIA RTX 3060 (12GB VRAM) or compatible
* **CUDA**: 11.x / 12.x
* **TensorRT**: 8.x / 9.x
* **CMake**: 3.16+
* **Compiler**: GCC 9+ / Clang 10+ with C++17 support
