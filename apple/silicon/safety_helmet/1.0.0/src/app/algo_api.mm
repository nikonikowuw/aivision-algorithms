/**
 * @file algo_api.mm
 * @brief Apple Silicon C-ABI entry point — ObjC++ variant with NSException interop.
 *
 * ## Role
 *
 * This is the external C ABI contract between the Go Engine host and the
 * safety_helmet algorithm shared library (tentcoo_detection.so). Every
 * exported symbol here is resolved via dlsym() by the Engine.
 *
 * ## Difference from algo_api.cpp
 *
 *   - algo_api.mm  — Apple Silicon: C++ try/catch + ObjC @try/@catch for
 *                     NSException interop. Supports HW_BUFFER_TYPE_APPLE_NATIVE
 *                     (zero-copy CVPixelBuffer) and Metal GPU preprocessing.
 *   - algo_api.cpp — Non-Apple fallback: plain C++ try/catch only. CPU
 *                     BGR24/NV12 paths only.
 *
 * ## Exported ABI symbols
 *
 *   detector_init      — Parse config, load model, return opaque handle.
 *   detector_infer     — Single-frame synchronous inference.
 *   detector_destroy   — Destroy handle, release GPU/memory resources.
 *   detector_version   — Return "1.0.0".
 *   detector_name      — Return "safety_helmet".
 *   detector_self_test — Self-check: load testimage.jpg, run inference.
 *   algo_free_result   — Free result_json allocated by detector_infer.
 *   detector_free_result — Convenience alias for algo_free_result.
 *
 * ## Exception safety (triple protection)
 *
 *   try {
 *       @try {
 *           // business logic
 *       } @catch (NSException *) { ... }   // Apple framework exceptions
 *   } catch (std::exception&) { ... }       // C++ standard exceptions
 *   catch (...) { ... }                     // Unknown C++ exceptions
 *
 * This three-layer guard prevents any exception from crossing the C-ABI
 * boundary into the Go host, which would cause a process panic.
 */

#include "pipeline/safety_helmet_pipeline.h"
#include "common/input_buffer_validator.h"
#include "common/logger.h"
#include "common/timer.h"
#include "algo/abi_contract.h"
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <exception>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <iomanip>
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// SerializeDetections — converts Detection vector to Engine-compliant JSON.
//
// Output schema (array of objects):
//   [
//     {
//       "category_code": 10001,
//       "category_name": "safety_hat",
//       "detect_confidence": 0.8765,
//       "bbox": [x, y, w, h]     // all in [0.0, 1.0], relative to original frame
//     },
//     ...
//   ]
// ---------------------------------------------------------------------------
static std::string SerializeDetections(const std::vector<safety_helmet::Detection>& detections) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6) << "[\n";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        ss << "  {\n";
        ss << "    \"category_code\": " << det.category_code << ",\n";
        ss << "    \"category_name\": \"" << det.category_name << "\",\n";
        ss << "    \"detect_confidence\": " << std::setprecision(4) << det.confidence << ",\n";
        ss << "    \"bbox\": [" 
           << det.x << ", " 
           << det.y << ", " 
           << det.w << ", " 
           << det.h << "]\n";
        ss << "  }" << (i + 1 < detections.size() ? "," : "") << "\n";
    }
    ss << "]";
    return ss.str();
}

// ---------------------------------------------------------------------------
// GetLibraryDir — self-discover the directory containing this .so.
//
// Uses dladdr() to find the runtime path of a symbol in this library,
// then extracts the directory component. This allows the algorithm to
// locate neighbouring files (model weights, test image) without hardcoded
// paths or environment variables.
// ---------------------------------------------------------------------------
static std::string GetLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&GetLibraryDir), &info) != 0 && info.dli_fname) {
        std::string path(info.dli_fname);
        auto last_slash = path.find_last_of('/');
        if (last_slash == std::string::npos) return ".";
        return path.substr(0, last_slash);
    }
    return ".";
}

/** Validates the borrowed CVPixelBuffer before any pipeline access. */
static bool ValidateAppleNativeDescriptor(const hw_buffer_desc_t& input,
                                          CVPixelBufferRef& pixel_buffer,
                                          std::string& error) {
    pixel_buffer = nullptr;
    error.clear();

    if (input.buffer_type != HW_BUFFER_TYPE_APPLE_NATIVE ||
        input.buffer_owner != HW_BUFFER_OWNER_ENGINE) {
        error = "Apple input must be Engine-owned APPLE_NATIVE memory";
        return false;
    }

    const hw_buffer_apple_t& apple = input.plat.apple;
    if (apple.abi_version != HW_BUFFER_APPLE_ABI_VERSION ||
        apple.struct_size != sizeof(hw_buffer_apple_t)) {
        error = "Apple native ABI version or struct size mismatch";
        return false;
    }
    if (apple.buffer_kind != HW_BUFFER_APPLE_CVPIXELBUFFER) {
        error = apple.buffer_kind == HW_BUFFER_APPLE_IOSURFACE
                    ? "direct IOSurface input is not supported"
                    : "Apple native input is not a CVPixelBuffer";
        return false;
    }
    if (apple.synchronization != 0u || apple.native_handle == 0u) {
        error = "Apple native synchronization or handle metadata is invalid";
        return false;
    }
    if (input.width == 0u || input.height == 0u ||
        input.width > safety_helmet::kMaxFrameDimension ||
        input.height > safety_helmet::kMaxFrameDimension) {
        error = "Apple native dimensions are outside the supported range";
        return false;
    }

    pixel_buffer = reinterpret_cast<CVPixelBufferRef>(
        static_cast<uintptr_t>(apple.native_handle));
    const size_t actual_width = CVPixelBufferGetWidth(pixel_buffer);
    const size_t actual_height = CVPixelBufferGetHeight(pixel_buffer);
    if (actual_width != input.width || actual_height != input.height) {
        error = "CVPixelBuffer dimensions do not match the descriptor";
        pixel_buffer = nullptr;
        return false;
    }

    const OSType actual_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
    if (actual_format != kCVPixelFormatType_32BGRA) {
        error = "only BGRA CVPixelBuffer input is supported";
        pixel_buffer = nullptr;
        return false;
    }
    if (input.pixel_format != actual_format || apple.pixel_format != actual_format) {
        error = "CVPixelBuffer FourCC does not match descriptor metadata";
        pixel_buffer = nullptr;
        return false;
    }

    const bool planar = CVPixelBufferIsPlanar(pixel_buffer);
    const size_t actual_plane_count = CVPixelBufferGetPlaneCount(pixel_buffer);
    if (planar || actual_plane_count != 0u || apple.plane_count != 1u) {
        error = "BGRA CVPixelBuffer plane count does not match the payload";
        pixel_buffer = nullptr;
        return false;
    }

    const size_t actual_stride = CVPixelBufferGetBytesPerRow(pixel_buffer);
    const size_t minimum_stride = actual_width * 4u;
    if (actual_stride < minimum_stride ||
        actual_stride > std::numeric_limits<uint32_t>::max() ||
        input.stride != actual_stride ||
        apple.plane_stride[0] != actual_stride ||
        apple.plane_stride[1] != 0u ||
        apple.plane_offset[0] != 0u || apple.plane_offset[1] != 0u) {
        error = "BGRA CVPixelBuffer stride or offset metadata mismatch";
        pixel_buffer = nullptr;
        return false;
    }

    return true;
}

extern "C" {

// ===================================================================
// C-ABI: detector_init
// ===================================================================
/**
 * @brief Initialise the safety helmet detection algorithm.
 *
 * Parses the JSON configuration string, loads the ONNX model via
 * ONNXRuntime (CoreML EP on Apple Silicon), and returns an opaque
 * handle for subsequent inference calls.
 *
 * @param config_json  JSON string with keys:
 *                     - "package_dir"  (string, optional)
 *                     - "model_path"   (string, optional, default "weights/...")
 *                     - "conf_threshold" (number, optional, default 0.5)
 *                     - "iou_threshold"  (number, optional, default 0.45)
 *
 * @return Non-null opaque handle on success.
 * @retval nullptr If config_json is null, model fails to load,
 *                 or any exception is caught (logged internally).
 */
algo_handle_t detector_init(const char *config_json) {
    try {
        @try {
            // Guard: config_json must be a valid C string.
            if (!config_json) return nullptr;
            auto pipeline = std::make_unique<safety_helmet::SafetyHelmetPipeline>();
            if (!pipeline->Initialize(config_json)) {
                return nullptr;
            }
            return reinterpret_cast<algo_handle_t>(pipeline.release());
        } @catch (NSException *nsException) {
            ALGO_LOG_ERROR("detector_init NSException: %s",
                           [[nsException description] UTF8String]);
            return nullptr;
        }
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_init C++ exception: %s", e.what());
        return nullptr;
    } catch (...) {
        ALGO_LOG_ERROR("detector_init unknown C++ exception");
        return nullptr;
    }
}

// ===================================================================
// C-ABI: detector_infer
// ===================================================================
/**
 * @brief Run single-frame safety helmet detection.
 *
 * Two input paths are supported:
 *   1. **CPU buffer path** (buffer_type != HW_BUFFER_TYPE_APPLE_NATIVE):
 *      Raw BGR24 or NV12 memory via input->data. Frame is wrapped as
 *      cv::Mat. NV12 is converted to BGR via cv::cvtColor.
 *      Preprocessing uses Metal GPU or CPU fallback.
 *   2. **Apple native buffer path** (buffer_type == HW_BUFFER_TYPE_APPLE_NATIVE):
 *      Borrowed CVPixelBufferRef from input->plat.apple.native_handle.
 *      Preprocessing reads directly from GPU memory via
 *      CVMetalTextureCache. Supported format: BGRA only.
 *
 * @param handle        Opaque handle from detector_init().
 * @param input         hw_buffer_desc_t with frame data.
 *                      - CPU path: input->data, width, height, stride, pixel_format.
 *                      - Apple native path: input->plat.apple (native_handle, buffer_kind, pixel_format).
 * @param context_json  Reserved for multi-algorithm chaining (unused, may be null).
 * @param result        [out] Struct filled with:
 *                      - result_json: malloc'd JSON string (caller frees via algo_free_result).
 *                      - result_json_len: byte length (excluding null terminator).
 *                      - infer_time_us: wall-clock inference time in microseconds.
 *
 * @return 0 on success.
 * @retval -1 null handle/input/result.
 * @retval -2 invalid input parameters (bad dimensions, unsupported format).
 * @retval -3 detection pipeline failed.
 * @retval -4 result allocation failed.
 * @retval -5 C++ exception or NSException caught.
 */
int detector_infer(algo_handle_t handle, const hw_buffer_desc_t *input,
                   const char *context_json, infer_result_t *result) {
    try {
        @try {
            // Parameter validation: handle, input, and result must all be non-null.
            if (!handle || !input || !result) return -1;
            (void)context_json;
            result->result_json = nullptr;
            result->result_json_len = 0;
            result->infer_time_us = 0;

            CVPixelBufferRef pixel_buffer = nullptr;
            safety_helmet::CpuBufferLayout cpu_layout;
            std::string validation_error;
            if (input->buffer_type == HW_BUFFER_TYPE_APPLE_NATIVE) {
                if (!ValidateAppleNativeDescriptor(
                        *input, pixel_buffer, validation_error)) {
                    ALGO_LOG_ERROR("detector_infer: %s", validation_error.c_str());
                    return -2;
                }
            } else if (!safety_helmet::ValidateCpuBufferDescriptor(
                           *input, cpu_layout, validation_error)) {
                ALGO_LOG_ERROR("detector_infer: %s", validation_error.c_str());
                return -2;
            }

            safety_helmet::Timer timer;
            timer.start();

            auto *pipeline = reinterpret_cast<safety_helmet::SafetyHelmetPipeline *>(handle);
            std::vector<safety_helmet::Detection> detections;

            // Apple native input is accepted only after strict CVPixelBuffer validation.
            if (pixel_buffer) {
                if (!pipeline->DetectFromPixelBuffer(pixel_buffer,
                                                      static_cast<int>(input->width),
                                                      static_cast<int>(input->height),
                                                      input->pixel_format,
                                                      detections)) {
                    ALGO_LOG_ERROR("detector_infer: Metal detection failed");
                    return -3;
                }
            } else {
                // ── Legacy CPU buffer path (BGR24 / NV12 raw memory) ──
                cv::Mat frame;
                int frame_w = static_cast<int>(input->width);
                int frame_h = static_cast<int>(input->height);

                if (input->pixel_format == safety_helmet::pixel_format::BGR24) {
                    // Direct wrap — cv::Mat aliases input->data, no copy.
                    // Caller must ensure data outlives this call (synchronous, so safe).
                    frame = cv::Mat(frame_h, frame_w, CV_8UC3,
                                    const_cast<void*>(input->data), input->stride);
                } else if (input->pixel_format == safety_helmet::pixel_format::NV12) {
                    // NV12 semiplanar: Y plane at offset 0 (W×H), interleaved UV at
                    // offset W×H (½W × ½H). Total buffer: H × 3/2 rows.
                    cv::Mat nv12_mat(frame_h * 3 / 2, frame_w, CV_8UC1,
                                     const_cast<void*>(input->data), input->stride);
                    cv::Mat bgr_mat;
                    cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);
                    // Crop to actual frame dimensions (conversion may produce padded output).
                    frame = bgr_mat(cv::Rect(0, 0, frame_w, frame_h));
                } else {
                    ALGO_LOG_ERROR("detector_infer: Unsupported pixel format: 0x%08x",
                                   input->pixel_format);
                    return -2;
                }

                if (!pipeline->Detect(frame, detections)) {
                    ALGO_LOG_ERROR("detector_infer: Detection failed");
                    return -3;
                }
            }

            // Serialize detections to JSON and allocate result buffer.
            // The caller (Engine) will free result_json via algo_free_result().
            std::string json_str = SerializeDetections(detections);
            char *result_json = static_cast<char *>(std::malloc(json_str.size() + 1u));
            if (!result_json) {
                ALGO_LOG_ERROR("detector_infer: Result allocation failed");
                return -4;
            }
            std::memcpy(result_json, json_str.data(), json_str.size());
            result_json[json_str.size()] = '\0';
            result->result_json = result_json;
            result->result_json_len = json_str.size();
            result->infer_time_us = timer.elapsed_us();
            return 0;
        } @catch (NSException *nsException) {
            ALGO_LOG_ERROR("detector_infer NSException: %s",
                           [[nsException description] UTF8String]);
            return -5;
        }
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_infer C++ exception: %s", e.what());
        return -5;
    } catch (...) {
        ALGO_LOG_ERROR("detector_infer unknown C++ exception");
        return -5;
    }
}

// ===================================================================
// C-ABI: detector_destroy
// ===================================================================
/**
 * @brief Destroy algorithm handle and release all resources.
 *
 * Idempotent: passing nullptr is safe and does nothing.
 * After this call the handle is invalid and must not be reused.
 *
 * @param handle  Opaque handle from detector_init(), or nullptr.
 */
void detector_destroy(algo_handle_t handle) {
    try {
        @try {
            if (handle) {
                auto *pipeline = reinterpret_cast<safety_helmet::SafetyHelmetPipeline *>(handle);
                pipeline->Destroy();
                delete pipeline;
            }
        } @catch (NSException *nsException) {
            ALGO_LOG_ERROR("detector_destroy NSException: %s",
                           [[nsException description] UTF8String]);
        }
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_destroy C++ exception: %s", e.what());
    } catch (...) {
        ALGO_LOG_ERROR("detector_destroy unknown C++ exception");
    }
}

/**
 * @brief Return algorithm version string (SemVer).
 * @return Static C string, always non-null.
 */
const char *detector_version(void) {
    return "1.0.0";
}

/**
 * @brief Return algorithm identifier name.
 * @return Static C string, always non-null.
 */
const char *detector_name(void) {
    return "safety_helmet";
}

// ===================================================================
// C-ABI: detector_self_test
// ===================================================================
/**
 * @brief Self-test: loads testimage.jpg from the package directory,
 *        runs a full detect-infer-destroy cycle, and reports success.
 *
 * This is called by the platform's Asynq background task after package
 * upload. A non-zero return value marks the algorithm as failed and
 * prevents it from going online.
 *
 * Steps:
 *   1. Verify ABI struct sizes (hw_buffer_desc_t == 144, infer_result_t == 40).
 *   2. Locate testimage.jpg next to this .so (via GetLibraryDir).
 *   3. Construct a minimal config JSON with model_path and defaults.
 *   4. detector_init → detector_infer (BGR24 from cv::imread) → detector_destroy.
 *   5. Log the inference result JSON.
 *
 * @return 0 on success, negative error code on failure (see inline returns).
 * @retval -1 hw_buffer_desc_t size mismatch (ABI break).
 * @retval -2 infer_result_t size mismatch (ABI break).
 * @retval -3 testimage.jpg not found.
 * @retval -4 detector_init returned null.
 * @retval -5 cv::imread failed.
 * @retval -6 detector_infer failed.
 * @retval -8 C++ exception or NSException caught.
 */
int detector_self_test(void) {
    try {
        @try {
            // ABI struct size guards — these must match the Engine's definitions
            // exactly. A mismatch indicates a version skew between algorithm and engine.
            if (sizeof(hw_buffer_desc_t) != 144) {
                ALGO_LOG_ERROR("self_test failed: hw_buffer_desc_t size %zu != 144",
                               sizeof(hw_buffer_desc_t));
                return -1;
            }
            if (sizeof(infer_result_t) != 40) {
                ALGO_LOG_ERROR("self_test failed: infer_result_t size %zu != 40",
                               sizeof(infer_result_t));
                return -2;
            }

            std::string lib_dir = GetLibraryDir();
            std::string image_path = safety_helmet::JoinPath(lib_dir, "testimage.jpg");
            std::string model_path = safety_helmet::JoinPath(lib_dir,
                "weights/damoyolo_safety_helmet.onnx");

            if (!safety_helmet::FileExists(image_path)) {
                ALGO_LOG_ERROR("self_test failed: testimage.jpg not found at %s",
                               image_path.c_str());
                return -3;
            }

            // Build minimal config JSON — model path is relative to package dir.
            std::stringstream ss;
            ss << "{\n"
               << "  \"package_dir\": \"" << lib_dir << "\",\n"
               << "  \"model_path\": \"" << model_path << "\",\n"
               << "  \"conf_threshold\": 0.5,\n"
               << "  \"iou_threshold\": 0.45\n"
               << "}";

            std::string config_json = ss.str();
            
            algo_handle_t handle = detector_init(config_json.c_str());
            if (!handle) {
                ALGO_LOG_ERROR("self_test failed: detector_init returned null");
                return -4;
            }

            cv::Mat img = cv::imread(image_path);
            if (img.empty()) {
                ALGO_LOG_ERROR("self_test failed: cv::imread could not load %s",
                               image_path.c_str());
                detector_destroy(handle);
                return -5;
            }

            // Construct a CPU BGR24 buffer descriptor from the loaded image.
            hw_buffer_desc_t input{};
            input.width = img.cols;
            input.height = img.rows;
            input.pixel_format = safety_helmet::pixel_format::BGR24;
            input.data = img.data;
            input.stride = img.cols * 3;
            input.size = img.cols * img.rows * 3;

            infer_result_t result{};
            int ret = detector_infer(handle, &input, nullptr, &result);
            if (ret != 0) {
                ALGO_LOG_ERROR("self_test failed: detector_infer returned %d", ret);
                detector_destroy(handle);
                return -6;
            }

            ALGO_LOG_INFO("self_test: Inference succeeded. Results:\n%s",
                          result.result_json);
            algo_free_result(&result);

            detector_destroy(handle);
            ALGO_LOG_INFO("self_test passed successfully");
            return 0;
        } @catch (NSException *nsException) {
            ALGO_LOG_ERROR("self_test NSException: %s",
                           [[nsException description] UTF8String]);
            return -8;
        }
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("self_test C++ exception: %s", e.what());
        return -8;
    } catch (...) {
        ALGO_LOG_ERROR("self_test unknown C++ exception");
        return -8;
    }
}

/**
 * @brief Free memory allocated by detector_infer for result_json.
 *
 * Safe to call with nullptr or with result->result_json == nullptr.
 *
 * @param result  Pointer to the result struct returned by detector_infer.
 */
void algo_free_result(infer_result_t *result) {
    if (result && result->result_json) {
        std::free(result->result_json);
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}

/**
 * @brief Convenience alias for algo_free_result (legacy naming compatibility).
 */
void detector_free_result(infer_result_t *result) {
    algo_free_result(result);
}

} // extern "C"
