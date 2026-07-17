/**
 * @file safety_helmet_pipeline.h
 * @brief Core inference pipeline: preprocess → model forward → postprocess.
 *
 * ## Responsibility
 *
 * SafetyHelmetPipeline orchestrates the full DAMO-YOLO-S detection flow:
 *   1. Preprocessing   — Metal GPU letterbox (or CPU fallback).
 *   2. Inference       — ONNXRuntime CoreML EP via DamoYoloModel.
 *   3. Output decode   — DAMO-YOLO decoder (score thresholding).
 *   4. NMS             — Class-aware Non-Maximum Suppression.
 *   5. Coordinate map  — Reverse letterbox + normalise to [0,1].
 *
 * ## Thread safety
 *
 * All public methods acquire a std::mutex, serialising concurrent access.
 * One pipeline instance per detector_init() call; each C-ABI handle owns
 * its own pipeline.
 *
 * ## GPU preprocessing
 *
 * On Apple Silicon with Metal available, preprocessing runs on the GPU via
 * MetalPreprocessor. When Metal is unavailable (or on non-Apple platforms),
 * the pipeline transparently falls back to CPU-based LetterBoxPreprocess().
 */

#ifndef SAFETY_HELMET_PIPELINE_H
#define SAFETY_HELMET_PIPELINE_H

#include "common/config_parser.h"
#include "models/damoyolo_model.h"
#include "postprocess/detection_result.h"
#include "preprocess/metal_preprocessor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <memory>

namespace safety_helmet {

class SafetyHelmetPipeline {
public:
    SafetyHelmetPipeline() = default;
    ~SafetyHelmetPipeline() = default;

    /**
     * @brief Initialise model, Metal preprocessor, and config.
     *
     * Loads the ONNX model via ONNXRuntime (CoreML EP on Apple Silicon),
     * initialises the Metal preprocessor if available, and parses runtime
     * configuration from the JSON string.
     *
     * @param config_json  JSON config with keys: package_dir, model_path,
     *                     conf_threshold, iou_threshold.
     * @return true on success, false if model load or config parse fails.
     */
    bool Initialize(const char* config_json);

    /**
     * @brief Run detection on a BGR cv::Mat frame.
     *
     * Orchestrates the full pipeline. Preprocessing uses Metal GPU when
     * available, falling back to CPU LetterBoxPreprocess.
     *
     * @param frame   Input BGR image (CV_8UC3), non-empty.
     * @param results [out] Detection vectors with normalised [0,1] bbox.
     * @return true on success.
     */
    bool Detect(const cv::Mat& frame, std::vector<Detection>& results);

    /**
     * @brief Zero-copy detection from a borrowed CVPixelBuffer.
     *
     * Used when the Engine provides HW_BUFFER_TYPE_METAL. The pixel buffer
     * is accessed via CVMetalTextureCache — no intermediate cv::Mat or CPU
     * copy. Preprocessing runs entirely on GPU.
     *
     * @param cvPixelBuffer  Borrowed CVPixelBufferRef, cast to void*.
     *                       Must outlive this call; must NOT be retained/released.
     * @param width          Pixel buffer width in pixels.
     * @param height         Pixel buffer height in pixels.
     * @param pixelFormat    CoreVideo FourCC (e.g. kCVPixelFormatType_32BGRA).
     * @param results        [out] Detection vectors with normalised [0,1] bbox.
     * @return true on success, false if Metal unavailable or format unsupported.
     */
    bool DetectFromPixelBuffer(void* cvPixelBuffer, int width, int height,
                               uint32_t pixelFormat, std::vector<Detection>& results);

    /**
     * @brief Destroy pipeline resources (model, Metal preprocessor).
     *
     * Sets initialized_ = false. The enclosing detector_destroy() is
     * responsible for deleting the pipeline object itself.
     */
    void Destroy();

    /** @return Immutable reference to parsed configuration. */
    const AlgoConfig& GetConfig() const { return config_; }

private:
    AlgoConfig config_;                                     ///< Parsed runtime config
    DamoYoloModel model_;                                   ///< ONNX Runtime model wrapper
    std::unique_ptr<MetalPreprocessor> metal_preprocessor_; ///< GPU preprocessing (or null)
    bool initialized_ = false;
    std::mutex mutex_;                                      ///< Serialises Detect() calls
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_PIPELINE_H
