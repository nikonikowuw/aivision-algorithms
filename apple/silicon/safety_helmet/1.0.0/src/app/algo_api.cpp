/**
 * @file algo_api.cpp
 * @brief Non-Apple C-ABI entry point — plain C++ variant (no ObjC).
 *
 * This is the CPU-only variant of the ABI entry point, used on Linux/Windows
 * builds where Metal and NSException are unavailable. The Apple Silicon
 * variant (algo_api.mm) provides the same interface plus NSException interop
 * and HW_BUFFER_TYPE_METAL support.
 *
 * ## Exported ABI symbols
 *
 *   detector_init      — Parse config, load model, return opaque handle.
 *   detector_infer     — Single-frame synchronous inference (BGR24/NV12 CPU paths).
 *   detector_destroy   — Destroy handle, release resources.
 *   detector_version   — Return "1.0.0".
 *   detector_name      — Return "safety_helmet".
 *   detector_self_test — Self-check with testimage.jpg.
 *   algo_free_result   — Free result_json.
 *   detector_free_result — Alias for algo_free_result.
 *
 * ## Exception safety
 *
 * All C-ABI functions are wrapped in C++ try/catch. On non-Apple platforms
 * CoreML is not loaded, so NSException is not a concern.
 */

#include "pipeline/safety_helmet_pipeline.h"
#include "common/input_buffer_validator.h"
#include "common/logger.h"
#include "common/timer.h"
#include "algo/abi_contract.h"
#include <exception>
#include <cstring>
#include <cstdlib>
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
// GetLibraryDir — self-discover the directory containing this .so via dladdr.
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

extern "C" {

/**
 * @brief Initialise the safety helmet detection algorithm.
 *
 * @param config_json  JSON string with keys: package_dir, model_path,
 *                     conf_threshold, iou_threshold.
 * @return Non-null opaque handle on success, nullptr on failure.
 */
algo_handle_t detector_init(const char *config_json) {
    try {
        if (!config_json) return nullptr;
        auto pipeline = std::make_unique<safety_helmet::SafetyHelmetPipeline>();
        if (!pipeline->Initialize(config_json)) {
            return nullptr;
        }
        return reinterpret_cast<algo_handle_t>(pipeline.release());
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_init C++ exception: %s", e.what());
        return nullptr;
    } catch (...) {
        ALGO_LOG_ERROR("detector_init unknown C++ exception");
        return nullptr;
    }
}

/**
 * @brief Run single-frame safety helmet detection (CPU path only).
 *
 * Supports BGR24 (direct wrap) and NV12 (YUV→BGR via cv::cvtColor).
 *
 * @param handle        Opaque handle from detector_init().
 * @param input         hw_buffer_desc_t with frame data (data, width, height,
 *                      stride, pixel_format).
 * @param context_json  Reserved for multi-algorithm chaining (unused).
 * @param result        [out] result_json (malloc'd), result_json_len, infer_time_us.
 * @return 0 on success, negative error code on failure.
 */
int detector_infer(algo_handle_t handle, const hw_buffer_desc_t *input,
                   const char *context_json, infer_result_t *result) {
    try {
        (void)context_json;  // Reserved for multi-algorithm chaining.
        if (!handle || !input || !result) return -1;
        result->result_json = nullptr;
        result->result_json_len = 0;
        result->infer_time_us = 0;

        safety_helmet::CpuBufferLayout layout;
        std::string validation_error;
        if (!safety_helmet::ValidateCpuBufferDescriptor(
                *input, layout, validation_error)) {
            ALGO_LOG_ERROR("detector_infer: %s", validation_error.c_str());
            return -2;
        }

        safety_helmet::Timer timer;
        timer.start();

        auto *pipeline = reinterpret_cast<safety_helmet::SafetyHelmetPipeline *>(handle);

        cv::Mat frame;
        int frame_w = static_cast<int>(input->width);
        int frame_h = static_cast<int>(input->height);

        if (input->pixel_format == safety_helmet::pixel_format::BGR24) {
            // Direct wrap — cv::Mat aliases input->data, no copy.
            frame = cv::Mat(frame_h, frame_w, CV_8UC3, const_cast<void*>(input->data), input->stride);
        } else if (input->pixel_format == safety_helmet::pixel_format::NV12) {
            // NV12 semiplanar: Y plane (W×H) + interleaved UV (½W×½H).
            // Total rows = H * 3/2, stride = input->stride.
            cv::Mat nv12_mat(frame_h * 3 / 2, frame_w, CV_8UC1,
                             const_cast<void*>(input->data), input->stride);
            cv::Mat bgr_mat;
            cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12);
            // Crop to actual dimensions — cvtColor may produce padded output.
            frame = bgr_mat(cv::Rect(0, 0, frame_w, frame_h));
        } else {
            ALGO_LOG_ERROR("detector_infer: Unsupported pixel format: 0x%08x", input->pixel_format);
            return -2;
        }

        std::vector<safety_helmet::Detection> detections;
        if (!pipeline->Detect(frame, detections)) {
            ALGO_LOG_ERROR("detector_infer: Detection failed");
            return -3;
        }

        // Serialize to JSON and allocate result buffer.
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
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_infer C++ exception: %s", e.what());
        return -5;
    } catch (...) {
        ALGO_LOG_ERROR("detector_infer unknown C++ exception");
        return -5;
    }
}

/**
 * @brief Destroy algorithm handle and release all resources.
 *
 * Idempotent: nullptr is safe.
 *
 * @param handle  Opaque handle from detector_init(), or nullptr.
 */
void detector_destroy(algo_handle_t handle) {
    try {
        if (handle) {
            auto *pipeline = reinterpret_cast<safety_helmet::SafetyHelmetPipeline *>(handle);
            pipeline->Destroy();
            delete pipeline;
        }
    } catch (const std::exception &e) {
        ALGO_LOG_ERROR("detector_destroy C++ exception: %s", e.what());
    } catch (...) {
        ALGO_LOG_ERROR("detector_destroy unknown C++ exception");
    }
}

/** @return Static version string "1.0.0". */
const char *detector_version(void) {
    return "1.0.0";
}

/** @return Static algorithm name "safety_helmet". */
const char *detector_name(void) {
    return "safety_helmet";
}

/**
 * @brief Self-test: load testimage.jpg, run detect-infer-destroy, report result.
 *
 * Verifies ABI struct sizes, loads the test image from the package directory,
 * and runs a full inference cycle. Called by the platform's Asynq background
 * task after package upload.
 *
 * @return 0 on success, negative error code on failure.
 * @retval -1/-2 ABI struct size mismatch.
 * @retval -3 testimage.jpg not found.
 * @retval -4 detector_init failed.
 * @retval -5 cv::imread failed.
 * @retval -6 detector_infer failed.
 * @retval -8 Exception caught.
 */
int detector_self_test(void) {
    try {
        // ABI struct size guards — must match the Engine's definitions.
        if (sizeof(hw_buffer_desc_t) != 144) {
            ALGO_LOG_ERROR("self_test failed: hw_buffer_desc_t size %zu != 144", sizeof(hw_buffer_desc_t));
            return -1;
        }
        if (sizeof(infer_result_t) != 40) {
            ALGO_LOG_ERROR("self_test failed: infer_result_t size %zu != 40", sizeof(infer_result_t));
            return -2;
        }

        std::string lib_dir = GetLibraryDir();
        std::string image_path = safety_helmet::JoinPath(lib_dir, "testimage.jpg");
        std::string model_path = safety_helmet::JoinPath(lib_dir, "weights/damoyolo_safety_helmet.onnx");

        if (!safety_helmet::FileExists(image_path)) {
            ALGO_LOG_ERROR("self_test failed: testimage.jpg not found at %s", image_path.c_str());
            return -3;
        }

        // Build minimal config JSON.
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
            ALGO_LOG_ERROR("self_test failed: cv::imread could not load %s", image_path.c_str());
            detector_destroy(handle);
            return -5;
        }

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

        ALGO_LOG_INFO("self_test: Inference succeeded. Results:\n%s", result.result_json);
        algo_free_result(&result);

        detector_destroy(handle);
        ALGO_LOG_INFO("self_test passed successfully");
        return 0;
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
 * Safe to call with nullptr or result->result_json == nullptr.
 *
 * @param result  Pointer to the result struct.
 */
void algo_free_result(infer_result_t *result) {
    if (result && result->result_json) {
        std::free(result->result_json);
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}

/** @brief Convenience alias for algo_free_result(). */
void detector_free_result(infer_result_t *result) {
    algo_free_result(result);
}

} // extern "C"
