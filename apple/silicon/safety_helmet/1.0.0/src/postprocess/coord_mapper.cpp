#include "coord_mapper.h"
#include <algorithm>

namespace safety_helmet {

Detection MapToOriginal(const RawDetection& raw_det, const LetterBoxInfo& info) {
    Detection det;
    det.confidence = raw_det.score;

    if (raw_det.class_id == 0) {
        det.category_code = 10001;
        det.category_name = "safety_hat";
    } else {
        det.category_code = 10002;
        det.category_name = "no_safety_hat";
    }

    // 1. Remove letterbox padding
    float x1_unpad = raw_det.x1 - info.pad_x;
    float y1_unpad = raw_det.y1 - info.pad_y;
    float x2_unpad = raw_det.x2 - info.pad_x;
    float y2_unpad = raw_det.y2 - info.pad_y;

    // 2. Scale back to original image coordinates
    float x1_orig = x1_unpad / info.ratio;
    float y1_orig = y1_unpad / info.ratio;
    float x2_orig = x2_unpad / info.ratio;
    float y2_orig = y2_unpad / info.ratio;

    // Clip to original image boundaries
    x1_orig = std::max(0.0f, std::min(x1_orig, static_cast<float>(info.orig_w)));
    y1_orig = std::max(0.0f, std::min(y1_orig, static_cast<float>(info.orig_h)));
    x2_orig = std::max(0.0f, std::min(x2_orig, static_cast<float>(info.orig_w)));
    y2_orig = std::max(0.0f, std::min(y2_orig, static_cast<float>(info.orig_h)));

    float w_orig = std::max(0.0f, x2_orig - x1_orig);
    float h_orig = std::max(0.0f, y2_orig - y1_orig);

    // 3. Normalize coordinates relative to original width and height
    det.x = x1_orig / info.orig_w;
    det.y = y1_orig / info.orig_h;
    det.w = w_orig / info.orig_w;
    det.h = h_orig / info.orig_h;

    // Ensure they are strictly within [0.0, 1.0]
    det.x = std::max(0.0f, std::min(det.x, 1.0f));
    det.y = std::max(0.0f, std::min(det.y, 1.0f));
    det.w = std::max(0.0f, std::min(det.w, 1.0f - det.x));
    det.h = std::max(0.0f, std::min(det.h, 1.0f - det.y));

    return det;
}

} // namespace safety_helmet
