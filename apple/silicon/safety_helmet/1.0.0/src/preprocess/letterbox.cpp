/**
 * @file letterbox.cpp
 * @brief CPU letterbox preprocessing implementation.
 *
 * This is the CPU fallback path. On Apple Silicon, MetalPreprocessor provides
 * a GPU-accelerated alternative that fuses all five passes into one GPU dispatch.
 */

#include "letterbox.h"

namespace safety_helmet {

/**
 * @brief CPU-based letterbox: resize + pad + BGR→RGB + normalise.
 *
 * ### Maths
 *
 *   ratio = min(target / W, target / H)
 *   newW  = round(W * ratio)
 *   newH  = round(H * ratio)
 *   padX  = (target - newW) / 2
 *   padY  = (target - newH) / 2
 *
 * The +-0.1f trick in round(dh ± 0.1) is the standard YOLO convention for
 * asymmetric padding: rounding half-away-from-zero ensures the padded area
 * is evenly distributed to within 1 pixel.
 *
 * Padding colour (114, 114, 114) is the standard YOLO grayscale mean.
 *
 * ### Output layout
 *
 * Channel-first (CHW): output[c*H*W + y*W + x] where c=0→R, 1→G, 2→B.
 * This matches ONNXRuntime's NCHW expectation: [1, 3, 640, 640].
 */
bool LetterBoxPreprocess(const cv::Mat& src, cv::Mat& dst,
                         std::vector<float>& input_tensor,
                         int target_size, LetterBoxInfo& info) {
    // Guard against empty input.
    if (src.empty()) {
        return false;
    }

    info.orig_w = src.cols;
    info.orig_h = src.rows;

    // ── 1. Compute scale factor preserving aspect ratio ──
    // The longer side is scaled to target_size; the shorter side gets padding.
    float r = std::min(static_cast<float>(target_size) / src.cols,
                       static_cast<float>(target_size) / src.rows);
    info.ratio = r;

    int new_unpad_w = static_cast<int>(std::round(src.cols * r));
    int new_unpad_h = static_cast<int>(std::round(src.rows * r));

    // ── 2. Bilinear resize ──
    cv::Mat tmp;
    if (src.cols != new_unpad_w || src.rows != new_unpad_h) {
        cv::resize(src, tmp, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        tmp = src.clone();  // Avoid aliasing when dimensions match exactly.
    }

    // ── 3. Compute padding ──
    float dw = static_cast<float>(target_size - new_unpad_w);
    float dh = static_cast<float>(target_size - new_unpad_h);

    dw /= 2.0f;
    dh /= 2.0f;

    info.pad_x = dw;
    info.pad_y = dh;

    // YOLO convention: round(dh - 0.1) for top, round(dh + 0.1) for bottom.
    // This distributes asymmetric padding with the extra pixel on bottom/right.
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    // Padding colour: (114, 114, 114) — standard YOLO grayscale.
    cv::copyMakeBorder(tmp, dst, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // ── 4. BGR → RGB channel swap ──
    // OpenCV loads images as BGR; the model expects RGB.
    cv::Mat rgb;
    cv::cvtColor(dst, rgb, cv::COLOR_BGR2RGB);

    // ── 5. Channel split and float normalisation ──
    // Split into R, G, B planes, convert each to float32, divide by 255.
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    input_tensor.resize(3 * target_size * target_size);
    float* ptr = input_tensor.data();

    for (int c = 0; c < 3; ++c) {
        // Each channel occupies a contiguous block: c * H * W.
        cv::Mat dest_channel(target_size, target_size, CV_32FC1,
                             ptr + c * target_size * target_size);
        channels[c].convertTo(dest_channel, CV_32FC1, 1.0 / 255.0);
    }

    return true;
}

} // namespace safety_helmet
