/**
 * @file letterbox.h
 * @brief CPU letterbox preprocessing for YOLO-family models.
 *
 * Letterboxing resizes an input image to fit a fixed square target size
 * (e.g. 640×640) while preserving the original aspect ratio. The image is
 * scaled so the longer side equals the target, then centred with gray
 * (114,114,114) padding on the shorter sides.
 *
 * The output is a channel-first (CHW) float32 tensor with values in [0,1],
 * suitable for ONNXRuntime NCHW input.
 */

#ifndef SAFETY_HELMET_LETTERBOX_H
#define SAFETY_HELMET_LETTERBOX_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace safety_helmet {

/**
 * @brief Metadata needed to map model-space coordinates back to the original image.
 *
 * Forward transform (image → letterbox):
 *   scaled_w = round(orig_w * ratio)
 *   scaled_h = round(orig_h * ratio)
 *   pad_x = (target - scaled_w) / 2
 *   pad_y = (target - scaled_h) / 2
 *
 * Reverse transform (letterbox → original, used by MapToOriginal):
 *   x_orig = (x_letterbox - pad_x) / ratio
 *   y_orig = (y_letterbox - pad_y) / ratio
 */
struct LetterBoxInfo {
    float ratio = 1.0f;   ///< Scale factor: min(target/w, target/h)
    float pad_x = 0.0f;   ///< Horizontal padding in pixels (left side)
    float pad_y = 0.0f;   ///< Vertical padding in pixels (top side)
    int orig_w = 0;       ///< Original image width
    int orig_h = 0;       ///< Original image height
};

/**
 * @brief CPU-based letterbox preprocessing.
 *
 * Pipeline (5 passes over the image data):
 *   1. cv::resize (bilinear)  — scale to fit target.
 *   2. cv::copyMakeBorder     — pad with gray (114,114,114).
 *   3. cv::cvtColor           — BGR→RGB channel swap.
 *   4. cv::split              — separate R, G, B planes.
 *   5. convertTo(float, 1/255) — normalise each plane to [0,1].
 *
 * @param src           Input BGR image (CV_8UC3), must be non-empty.
 * @param dst           [out] Intermediate letterbox image (BGR, CV_8UC3, target×target).
 * @param input_tensor  [out] Final float tensor, resized to 3*target².
 *                      Layout: CHW, RGB order, values in [0.0, 1.0].
 * @param target_size   Square output side length (e.g. 640 for DAMO-YOLO-S).
 * @param info          [out] Letterbox metadata for downstream coordinate recovery.
 * @return true on success, false if src is empty.
 */
bool LetterBoxPreprocess(const cv::Mat& src, cv::Mat& dst,
                         std::vector<float>& input_tensor,
                         int target_size, LetterBoxInfo& info);

} // namespace safety_helmet

#endif // SAFETY_HELMET_LETTERBOX_H
