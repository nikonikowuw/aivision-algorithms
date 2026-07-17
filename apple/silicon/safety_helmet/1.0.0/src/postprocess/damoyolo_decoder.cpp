/**
 * @file damoyolo_decoder.cpp
 * @brief DAMO-YOLO-S output decoding implementation.
 *
 * DAMO-YOLO-S uses a dual-score anchor format: each anchor i produces
 * [x1, y1, x2, y2, score_class0, score_class1]. The class with the higher
 * score wins for that anchor. This differs from standard YOLO which outputs
 * [x, y, w, h, obj, prob_class0, prob_class1, ...] for each anchor.
 */

#include "damoyolo_decoder.h"
#include "common/logger.h"

namespace safety_helmet {

bool DecodeOutputs(const std::vector<std::vector<float>>& output_tensors,
                   const std::vector<std::vector<int64_t>>& output_shapes,
                   float conf_threshold,
                   std::vector<RawDetection>& raw_detections) {
    raw_detections.clear();

    // Guard: at least one output tensor required.
    if (output_tensors.empty() || output_shapes.empty()) {
        ALGO_LOG_ERROR("DecodeOutputs: No output tensors");
        return false;
    }

    // Validate tensor shape: [1, N, 6].
    // Dimension 0 = batch (always 1), dim 2 = 6 values per anchor.
    const auto& shape = output_shapes[0];
    if (shape.size() != 3 || shape[0] != 1 || shape[2] != 6) {
        ALGO_LOG_ERROR("DecodeOutputs: Invalid output shape, expected [1, N, 6], "
                       "got [%lld, %lld, %lld]",
                       shape.size() > 0 ? shape[0] : 0,
                       shape.size() > 1 ? shape[1] : 0,
                       shape.size() > 2 ? shape[2] : 0);
        return false;
    }

    int64_t num_anchors = shape[1];
    const float* data = output_tensors[0].data();

    // Iterate anchors: 6 consecutive floats per anchor.
    // Layout: x1, y1, x2, y2, score_0, score_1.
    for (int64_t i = 0; i < num_anchors; ++i) {
        float x1 = data[i * 6 + 0];
        float y1 = data[i * 6 + 1];
        float x2 = data[i * 6 + 2];
        float y2 = data[i * 6 + 3];
        float score_0 = data[i * 6 + 4];   // class 0: safety_hat
        float score_1 = data[i * 6 + 5];   // class 1: no_safety_hat

        // Dual-score argmax: the higher score determines the class.
        float max_score = score_0;
        int class_id = 0;
        if (score_1 > score_0) {
            max_score = score_1;
            class_id = 1;
        }

        // Discard low-confidence anchors.
        if (max_score >= conf_threshold) {
            RawDetection det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.score = max_score;
            det.class_id = class_id;

            raw_detections.push_back(det);
        }
    }

    return true;
}

} // namespace safety_helmet
