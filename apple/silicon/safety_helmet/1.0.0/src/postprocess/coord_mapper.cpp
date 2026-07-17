/**
 * @file coord_mapper.cpp
 * @brief Coordinate mapping: letterbox space → original normalised space.
 *
 * The forward letterbox transform (in LetterBoxPreprocess) does:
 *   x_lb = x_orig * ratio + pad_x
 *   y_lb = y_orig * ratio + pad_y
 *
 * This file inverts that transform:
 *   x_orig = (x_lb - pad_x) / ratio
 *
 * Then normalises by dividing by the original frame dimensions to produce
 * coordinates in [0.0, 1.0].
 */

#include "coord_mapper.h"
#include <algorithm>

namespace safety_helmet {

Detection MapToOriginal(const RawDetection& raw_det, const LetterBoxInfo& info) {
    Detection det;
    det.confidence = raw_det.score;

    // ── Map internal class_id to platform category_code ──
    // 10001 = safety_hat (wearing helmet), 10002 = no_safety_hat (not wearing).
    // These codes must match label_map.json for the platform's DB mapping.
    if (raw_det.class_id == 0) {
        det.category_code = 10001;
        det.category_name = "safety_hat";
    } else {
        det.category_code = 10002;
        det.category_name = "no_safety_hat";
    }

    // ── 1. Reverse letterbox: remove padding, then unscale ──
    // Subtract padding first (because padding was added AFTER scaling),
    // then divide by ratio to recover original image coordinates.
    float x1_unpad = raw_det.x1 - info.pad_x;
    float y1_unpad = raw_det.y1 - info.pad_y;
    float x2_unpad = raw_det.x2 - info.pad_x;
    float y2_unpad = raw_det.y2 - info.pad_y;

    float x1_orig = x1_unpad / info.ratio;
    float y1_orig = y1_unpad / info.ratio;
    float x2_orig = x2_unpad / info.ratio;
    float y2_orig = y2_unpad / info.ratio;

    // ── 2. Clip to original frame boundaries ──
    // Floating-point imprecision or extreme padding can produce out-of-range values.
    x1_orig = std::max(0.0f, std::min(x1_orig, static_cast<float>(info.orig_w)));
    y1_orig = std::max(0.0f, std::min(y1_orig, static_cast<float>(info.orig_h)));
    x2_orig = std::max(0.0f, std::min(x2_orig, static_cast<float>(info.orig_w)));
    y2_orig = std::max(0.0f, std::min(y2_orig, static_cast<float>(info.orig_h)));

    // Width/height: std::max(0, ...) guards against reversed corners.
    float w_orig = std::max(0.0f, x2_orig - x1_orig);
    float h_orig = std::max(0.0f, y2_orig - y1_orig);

    // ── 3. Normalise to [0.0, 1.0] relative to original frame dimensions ──
    det.x = x1_orig / info.orig_w;
    det.y = y1_orig / info.orig_h;
    det.w = w_orig / info.orig_w;
    det.h = h_orig / info.orig_h;

    // ── 4. Final clamp — ensure strict [0,1] compliance ──
    // The upper bound for w is (1.0 - x) so that x + w ≤ 1.0 always holds.
    // Same for h and y.
    det.x = std::max(0.0f, std::min(det.x, 1.0f));
    det.y = std::max(0.0f, std::min(det.y, 1.0f));
    det.w = std::max(0.0f, std::min(det.w, 1.0f - det.x));
    det.h = std::max(0.0f, std::min(det.h, 1.0f - det.y));

    return det;
}

} // namespace safety_helmet
