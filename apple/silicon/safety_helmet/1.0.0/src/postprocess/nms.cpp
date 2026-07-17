#include "nms.h"
#include <algorithm>

namespace safety_helmet {

static float CalculateIoU(const RawDetection& a, const RawDetection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection_area;

    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

std::vector<RawDetection> ApplyNMS(const std::vector<RawDetection>& detections, float iou_threshold) {
    std::vector<RawDetection> results;
    if (detections.empty()) return results;

    std::vector<RawDetection> sorted_dets = detections;
    std::sort(sorted_dets.begin(), sorted_dets.end(), [](const RawDetection& a, const RawDetection& b) {
        return a.score > b.score;
    });

    std::vector<bool> keep(sorted_dets.size(), true);

    for (size_t i = 0; i < sorted_dets.size(); ++i) {
        if (!keep[i]) continue;
        results.push_back(sorted_dets[i]);

        for (size_t j = i + 1; j < sorted_dets.size(); ++j) {
            if (!keep[j]) continue;
            if (sorted_dets[i].class_id == sorted_dets[j].class_id) {
                float iou = CalculateIoU(sorted_dets[i], sorted_dets[j]);
                if (iou > iou_threshold) {
                    keep[j] = false;
                }
            }
        }
    }

    return results;
}

} // namespace safety_helmet
