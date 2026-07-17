/**
 * @file metal_preprocessor.h
 * @brief GPU-accelerated image preprocessing via Apple Metal compute shaders.
 *
 * Encapsulates Metal device lifetime, shader compilation, and dispatch for
 * the DAMO-YOLO-S input pipeline: letterbox resize (fit-to-square with gray
 * padding), BGR->RGB channel swap, and uint8-to-float32 normalization in a
 * single GPU compute pass. Two input paths are supported:
 *
 *   1. Raw cv::Mat BGR buffer  → uploaded to MTLBuffer → kernel → output.
 *   2. Borrowed CVPixelBufferRef → CVMetalTextureCache (zero-copy) → kernel → output.
 *
 * The caller (SafetyHelmetPipeline) transparently falls back to CPU-based
 * LetterBoxPreprocess() when Metal is unavailable — no caller-side branching
 * needed beyond checking IsAvailable().
 *
 * Design note: the header is deliberately Objective-C free. All Apple types
 * are opaque behind struct Impl, keeping this header includable from pure C++
 * translation units.
 */
#ifndef SAFETY_HELMET_METAL_PREPROCESSOR_H
#define SAFETY_HELMET_METAL_PREPROCESSOR_H

#include "letterbox.h"
#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include <cstdint>

namespace safety_helmet {

/**
 * @brief GPU-accelerated preprocessor using Metal compute shaders.
 *
 * On Apple Silicon (M-series), offloads the full DAMO-YOLO-S preprocessing
 * pipeline — letterbox resize, BGR→RGB channel swap, and float32 normalization
 * — to a single Metal compute kernel dispatch. The kernel performs manual
 * bilinear interpolation on raw 8-bit BGR pixels and writes a channel-first
 * (CHW) float tensor ready for ONNXRuntime consumption.
 *
 * ### Performance motivation
 * The CPU path calls cv::resize (bilinear), cv::cvtColor (BGR→RGB),
 * cv::copyMakeBorder (gray padding), cv::split, and per-channel
 * convertTo(float, 1/255). These are 5 passes over the image data. The Metal
 * kernel fuses all five into one GPU pass, leveraging the Unified Memory
 * Architecture to keep upload/download costs low.
 *
 * ### Thread safety
 * Each pipeline instance owns exactly one MetalPreprocessor. The Metal
 * command queue is serial by design — concurrent frames are serialised at the
 * pipeline's std::mutex boundary, so no additional GPU synchronisation is
 * required.
 */
class MetalPreprocessor {
public:
    /**
     * @brief Creates a Metal device, command queue, and compiles embedded shaders.
     *
     * On first construction, the embedded MSL (Metal Shading Language) source
     * is compiled into two compute pipeline state objects: one for raw buffer
     * input (cv::Mat path) and one for texture input (CVPixelBuffer path).
     * Failure is non-fatal — IsAvailable() returns false and the caller uses
     * the CPU fallback.
     */
    MetalPreprocessor();
    ~MetalPreprocessor();

    // Non-copyable, non-movable (owns id<MTLDevice> and friends).
    MetalPreprocessor(const MetalPreprocessor&) = delete;
    MetalPreprocessor& operator=(const MetalPreprocessor&) = delete;
    MetalPreprocessor(MetalPreprocessor&&) = delete;
    MetalPreprocessor& operator=(MetalPreprocessor&&) = delete;

    /**
     * @brief Whether Metal GPU preprocessing is available.
     *
     * Returns false when: no Metal-capable GPU, shader compilation failed,
     * or running on a non-Apple platform.
     *
     * @return true if the GPU path can be used.
     */
    bool IsAvailable() const;

    /**
     * @brief GPU-accelerated letterbox preprocessing from cv::Mat.
     *
     * Uploads the BGR pixel data to a MTLBuffer (zero-copy when the cv::Mat
     * is contiguous with natural stride), dispatches the letterbox kernel, and
     * downloads the float32 tensor.
     *
     * @param bgr_image  Input image, must be CV_8UC3, non-empty.
     * @param output_tensor  [out] Resized to 3 * target_size^2 floats.
     *                       Layout: CHW, RGB order, values in [0.0, 1.0].
     * @param target_size  Square output side length (e.g. 640).
     * @param box_info  [out] Letterbox scale, padding, and original dimensions
     *                  needed by MapToOriginal() for bbox coordinate recovery.
     * @return true on success, false if input is invalid or GPU command fails.
     */
    bool Process(const cv::Mat& bgr_image,
                 std::vector<float>& output_tensor,
                 int target_size,
                 LetterBoxInfo& box_info);

    /**
     * @brief Zero-copy preprocessing from a borrowed CVPixelBuffer.
     *
     * Used when the Engine passes HW_BUFFER_TYPE_METAL. The CVPixelBuffer's
     * GPU backing is accessed via CVMetalTextureCache — no upload needed.
     * Currently supports BGRA (kCVPixelFormatType_32BGRA) only.
     *
     * @param cvPixelBuffer  Borrowed CVPixelBufferRef, cast to void*.
     *                       Must outlive this call; the algorithm must NOT
     *                       retain or release it.
     * @param output_tensor  [out] Same layout as Process().
     * @param target_size  Square output side length.
     * @param box_info  [out] Letterbox metadata for coordinate mapping.
     * @return true on success, false if format unsupported or GPU error.
     */
    bool ProcessFromPixelBuffer(void* cvPixelBuffer,
                                std::vector<float>& output_tensor,
                                int target_size,
                                LetterBoxInfo& box_info);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_METAL_PREPROCESSOR_H
