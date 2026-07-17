/**
 * @file safety_helmet_pipeline.cpp
 * @brief Inference pipeline implementation.
 *
 * ## Data flow (per frame)
 *
 *   frame (cv::Mat BGR or CVPixelBuffer)
 *     │
 *     ▼
 *   ┌─────────────────────────┐
 *   │  Preprocess              │
 *   │  Metal (GPU) or CPU      │
 *   │  Letterbox 640×640       │
 *   │  BGR→RGB + norm [0,1]   │
 *   └──────────┬──────────────┘
 *              │ float tensor [1, 3, 640, 640]
 *              ▼
 *   ┌─────────────────────────┐
 *   │  DamoYoloModel::Forward  │
 *   │  ONNXRuntime CoreML EP   │
 *   └──────────┬──────────────┘
 *              │ raw output [1, N, 6]  (x1,y1,x2,y2,score_c0,score_c1)
 *              ▼
 *   ┌─────────────────────────┐
 *   │  DecodeOutputs           │
 *   │  Threshold by conf       │
 *   └──────────┬──────────────┘
 *              │ vector<RawDetection>
 *              ▼
 *   ┌─────────────────────────┐
 *   │  ApplyNMS                │
 *   │  Class-aware IoU filter  │
 *   └──────────┬──────────────┘
 *              │ vector<RawDetection> (filtered)
 *              ▼
 *   ┌─────────────────────────┐
 *   │  MapToOriginal           │
 *   │  Undo letterbox + norm   │
 *   └──────────┬──────────────┘
 *              │ vector<Detection>  (bbox in [0,1])
 *              ▼
 *            caller
 *
 * ## Error handling
 *
 * Each stage returns bool. On failure the pipeline returns false immediately;
 * the C-ABI layer (algo_api) converts this to a negative error code. No
 * exceptions propagate out of Detect() — they are caught and logged.
 */

#include "safety_helmet_pipeline.h"
#include "preprocess/letterbox.h"
#include "postprocess/damoyolo_decoder.h"
#include "postprocess/nms.h"
#include "postprocess/coord_mapper.h"
#include "common/logger.h"

namespace safety_helmet {

bool SafetyHelmetPipeline::Initialize(const char* config_json) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return true;

    try {
        config_ = AlgoConfig::LoadConfig(config_json);
        
        ALGO_LOG_INFO("Initializing pipeline. Model path: %s, Conf: %.2f, IoU: %.2f",
                      config_.model_path.c_str(), config_.conf_threshold, config_.iou_threshold);

        // Load ONNX model via ONNXRuntime (CoreML EP auto-enabled on Apple Silicon).
        if (!model_.Init(config_.model_path)) {
            ALGO_LOG_ERROR("Failed to initialize DAMO-YOLO model");
            return false;
        }

        // Initialise Metal GPU preprocessor for Apple Silicon UMA.
        // On non-Apple platforms this is a no-op stub (IsAvailable() → false).
        metal_preprocessor_ = std::make_unique<MetalPreprocessor>();
        if (metal_preprocessor_->IsAvailable()) {
            ALGO_LOG_INFO("GPU-accelerated preprocessing enabled.");
        } else {
            ALGO_LOG_INFO("Metal unavailable, using CPU preprocessing fallback.");
        }

        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("Pipeline initialization failed with exception: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("Pipeline initialization failed with unknown exception");
        return false;
    }
}

bool SafetyHelmetPipeline::Detect(const cv::Mat& frame, std::vector<Detection>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results.clear();

    if (!initialized_) {
        ALGO_LOG_ERROR("Pipeline not initialized");
        return false;
    }

    if (frame.empty()) {
        ALGO_LOG_ERROR("Frame is empty");
        return false;
    }

    try {
        // ── 1. Preprocess (Letterbox + float conversion) ──
        // GPU path when Metal is available; CPU LetterBoxPreprocess as fallback.
        std::vector<float> input_tensor;
        LetterBoxInfo box_info;
        bool preprocess_ok = false;

        if (metal_preprocessor_ && metal_preprocessor_->IsAvailable()) {
            preprocess_ok = metal_preprocessor_->Process(frame, input_tensor, 640, box_info);
        }
        if (!preprocess_ok) {
            // CPU fallback: cv::resize + cv::cvtColor + cv::copyMakeBorder + split.
            cv::Mat processed_img;
            preprocess_ok = LetterBoxPreprocess(frame, processed_img, input_tensor, 640, box_info);
        }
        if (!preprocess_ok) {
            ALGO_LOG_ERROR("Preprocessing failed");
            return false;
        }

        // ── 2. Inference ──
        // ONNXRuntime forward pass. Input shape matches DAMO-YOLO-S: [1, 3, 640, 640].
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        std::vector<std::vector<float>> output_tensors;
        std::vector<std::vector<int64_t>> output_shapes;
        if (!model_.Forward(input_tensor, input_shape, output_tensors, output_shapes)) {
            ALGO_LOG_ERROR("Model inference forward failed");
            return false;
        }

        // ── 3. Output decode ──
        // DAMO-YOLO outputs shape [1, N, 6]: (x1, y1, x2, y2, score_cls0, score_cls1).
        // Filter by conf_threshold, select max-score class.
        std::vector<RawDetection> raw_detections;
        if (!DecodeOutputs(output_tensors, output_shapes, config_.conf_threshold, raw_detections)) {
            ALGO_LOG_ERROR("Decoding model outputs failed");
            return false;
        }

        // ── 4. Non-Maximum Suppression ──
        // Class-aware: IoU is only computed between detections of the same class.
        // Sorted by score descending before suppression for deterministic output.
        std::vector<RawDetection> filtered_raw = ApplyNMS(raw_detections, config_.iou_threshold);

        // ── 5. Coordinate mapping ──
        // Reverse letterbox (remove padding, unscale) and normalise to [0,1]
        // relative to the original frame dimensions.
        for (const auto& raw_det : filtered_raw) {
            results.push_back(MapToOriginal(raw_det, box_info));
        }

        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("Detect exception: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("Detect unknown exception");
        return false;
    }
}

void SafetyHelmetPipeline::Destroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    metal_preprocessor_.reset();  // release GPU resources
    ALGO_LOG_INFO("Pipeline destroyed");
}

bool SafetyHelmetPipeline::DetectFromPixelBuffer(void* cvPixelBuffer,
                                                   int width, int height,
                                                   uint32_t pixelFormat,
                                                   std::vector<Detection>& results) {
    std::lock_guard<std::mutex> lock(mutex_);
    results.clear();

    if (!initialized_) {
        ALGO_LOG_ERROR("Pipeline not initialized");
        return false;
    }

    // Guard: CVPixelBuffer must be non-null with valid dimensions.
    if (!cvPixelBuffer || width <= 0 || height <= 0) {
        ALGO_LOG_ERROR("DetectFromPixelBuffer: invalid parameters");
        return false;
    }
    (void)pixelFormat;  // Used by Metal preprocessor internally for format dispatch.

    try {
        // ── 1. Preprocess from CVPixelBuffer via Metal (zero-copy) ──
        // CVMetalTextureCache maps the IOSurface backing directly — no upload.
        std::vector<float> input_tensor;
        LetterBoxInfo box_info;
        bool preprocess_ok = false;

        if (metal_preprocessor_ && metal_preprocessor_->IsAvailable()) {
            preprocess_ok = metal_preprocessor_->ProcessFromPixelBuffer(
                cvPixelBuffer, input_tensor, 640, box_info);
        }

        if (!preprocess_ok) {
            ALGO_LOG_ERROR("Metal CVPixelBuffer preprocessing failed "
                           "(Metal unavailable or unsupported pixel format)");
            return false;
        }

        // ── 2. Inference (shared path with Detect) ──
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        std::vector<std::vector<float>> output_tensors;
        std::vector<std::vector<int64_t>> output_shapes;
        if (!model_.Forward(input_tensor, input_shape, output_tensors, output_shapes)) {
            ALGO_LOG_ERROR("Model inference forward failed");
            return false;
        }

        // ── 3. Output decode ──
        std::vector<RawDetection> raw_detections;
        if (!DecodeOutputs(output_tensors, output_shapes, config_.conf_threshold, raw_detections)) {
            ALGO_LOG_ERROR("Decoding model outputs failed");
            return false;
        }

        // ── 4. NMS ──
        std::vector<RawDetection> filtered_raw = ApplyNMS(raw_detections, config_.iou_threshold);

        // ── 5. Coordinate mapping ──
        for (const auto& raw_det : filtered_raw) {
            results.push_back(MapToOriginal(raw_det, box_info));
        }

        return true;
    } catch (const std::exception& e) {
        ALGO_LOG_ERROR("DetectFromPixelBuffer exception: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOG_ERROR("DetectFromPixelBuffer unknown exception");
        return false;
    }
}

} // namespace safety_helmet
