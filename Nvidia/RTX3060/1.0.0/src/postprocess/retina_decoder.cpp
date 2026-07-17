/**
 * @file retina_decoder.cpp
 * @brief GPU Face Recognition — RetinaFace decoder implementation.
 *
 * RetinaFace raw output layout (per anchor):
 *   bbox regression (4) + landmark regression (10) + class score (1)
 *   Total = 15 values per anchor.
 *   Format: [dx1, dy1, dx2, dy2, lx0, ly0, lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4, score]
 */

#include "retina_decoder.h"
#include "common/geometry_utils.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace face_rec {

std::vector<DetectedObject> RetinaDecoder::Nms(const std::vector<DetectedObject>& dets,
                                                float iou_threshold) {
    if (dets.empty()) return {};

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

std::vector<DetectedObject> RetinaDecoder::Decode(const std::vector<float>& raw_output,
                                                   int orig_width, int orig_height,
                                                   int input_width, int input_height,
                                                   float conf_thres, float nms_iou_thres) {
    // RetinaFace output: [1, N, 15] or [N, 15]
    constexpr int kStride = 15;
    size_t num_anchors = raw_output.size() / kStride;
    if (num_anchors == 0) return {};

    float x_scale = static_cast<float>(orig_width) / input_width;
    float y_scale = static_cast<float>(orig_height) / input_height;

    std::vector<DetectedObject> candidates;
    candidates.reserve(num_anchors);

    for (size_t i = 0; i < num_anchors; ++i) {
        const float* row = raw_output.data() + i * kStride;

        float score = row[14];  // Class score
        if (score < conf_thres) continue;

        // Bounding box (already in input coordinates)
        float x1 = row[0] * x_scale;
        float y1 = row[1] * y_scale;
        float x2 = row[2] * x_scale;
        float y2 = row[3] * y_scale;

        // Clamp
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_height)));

        if (x2 <= x1 || y2 <= y1) continue;

        DetectedObject obj;
        obj.bbox = {x1, y1, x2 - x1, y2 - y1};
        obj.confidence = score;
        obj.label = CAT_KNOWN_FACE;  // Will be refined after recognition

        // Landmarks (5 points): left_eye, right_eye, nose, left_mouth, right_mouth
        for (int l = 0; l < 5; ++l) {
            obj.landmarks[l].x = row[4 + l * 2] * x_scale;
            obj.landmarks[l].y = row[5 + l * 2] * y_scale;
        }

        candidates.push_back(obj);
    }

    return Nms(candidates, nms_iou_thres);
}

std::vector<std::vector<DetectedObject>> RetinaDecoder::DecodeBatch(
    const std::vector<std::vector<float>>& raw_outputs,
    const std::vector<int>& orig_widths, const std::vector<int>& orig_heights,
    int input_width, int input_height,
    float conf_thres, float nms_iou_thres) {

    size_t batch_size = raw_outputs.size();
    std::vector<std::vector<DetectedObject>> results(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        results[i] = Decode(raw_outputs[i],
                            orig_widths[i], orig_heights[i],
                            input_width, input_height,
                            conf_thres, nms_iou_thres);
    }

    return results;
}

} // namespace face_rec
