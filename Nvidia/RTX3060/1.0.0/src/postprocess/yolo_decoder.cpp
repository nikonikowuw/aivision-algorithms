/**
 * @file yolo_decoder.cpp
 * @brief GPU Face Recognition — YOLO11n decoder + NMS implementation.
 */

#include "yolo_decoder.h"
#include "common/geometry_utils.h"
#include <algorithm>
#include <numeric>

namespace face_rec {

/**
 * @brief Greedy NMS (CPU) — equivalent to cv::dnn::NMSBoxes behavior.
 */
std::vector<DetectedObject> YoloDecoder::Nms(const std::vector<DetectedObject>& dets,
                                              float iou_threshold) {
    if (dets.empty()) return {};

    // Sort by confidence descending
    std::vector<int> indices(dets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&dets](int a, int b) {
        return dets[a].confidence > dets[b].confidence;
    });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<DetectedObject> result;

    for (size_t i = 0; i < indices.size(); ++i) {
        if (suppressed[indices[i]]) continue;
        result.push_back(dets[indices[i]]);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            if (IoU(dets[indices[i]].bbox, dets[indices[j]].bbox) > iou_threshold) {
                suppressed[indices[j]] = true;
            }
        }
    }

    return result;
}

std::vector<DetectedObject> YoloDecoder::Decode(const std::vector<float>& raw_output,
                                                 int orig_width, int orig_height,
                                                 float scale, int pad_x, int pad_y,
                                                 float conf_thres, float nms_iou_thres) {
    // YOLO11n output shape: [1, 8400, 84] or [8400, 84]
    // Each row: [x_center, y_center, w, h, class_scores...]
    // For person detection: only class 0 (person) with 80 classes → stride = 4 + 80 = 84
    constexpr int kNumClasses = 80;  // COCO classes
    constexpr int kStride = 4 + kNumClasses;

    size_t num_anchors = raw_output.size() / kStride;
    if (num_anchors == 0) return {};

    std::vector<DetectedObject> candidates;
    candidates.reserve(num_anchors);

    for (size_t i = 0; i < num_anchors; ++i) {
        const float* row = raw_output.data() + i * kStride;

        // Person class is index 0 in COCO
        float class_score = row[4];  // class 0: person
        if (class_score < conf_thres) continue;

        // Decode bbox from letterbox coordinates to original coordinates
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];

        // Remove letterbox padding and rescale
        float x1 = (cx - w / 2.0f - pad_x) / scale;
        float y1 = (cy - h / 2.0f - pad_y) / scale;
        float bw = w / scale;
        float bh = h / scale;

        // Clamp to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        bw = std::min(bw, static_cast<float>(orig_width) - x1);
        bh = std::min(bh, static_cast<float>(orig_height) - y1);

        DetectedObject obj;
        obj.bbox = {x1, y1, bw, bh};
        obj.confidence = class_score;
        obj.label = CAT_PERSON;
        candidates.push_back(obj);
    }

    return Nms(candidates, nms_iou_thres);
}

std::vector<std::vector<DetectedObject>> YoloDecoder::DecodeBatch(
    const std::vector<std::vector<float>>& raw_outputs,
    const std::vector<int>& orig_widths, const std::vector<int>& orig_heights,
    const std::vector<float>& scales,
    const std::vector<int>& pad_xs, const std::vector<int>& pad_ys,
    float conf_thres, float nms_iou_thres) {

    size_t batch_size = raw_outputs.size();
    std::vector<std::vector<DetectedObject>> results(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        results[i] = Decode(raw_outputs[i],
                            orig_widths[i], orig_heights[i],
                            scales[i], pad_xs[i], pad_ys[i],
                            conf_thres, nms_iou_thres);
    }

    return results;
}

} // namespace face_rec
