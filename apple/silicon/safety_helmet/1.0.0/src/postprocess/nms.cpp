/**
 * @file nms.cpp
 * @brief Class-aware Non-Maximum Suppression implementation.
 */

#include "nms.h"
#include <algorithm>

namespace safety_helmet {

/**
 * @brief Calculate Intersection-over-Union between two axis-aligned boxes.
 *
 * IoU = intersection_area / union_area.
 *
 * Division-by-zero guard: if union_area ≤ 0, returns 0.0 (boxes are disjoint
 * or degenerate — no suppression should occur).
 */
static float CalculateIoU(const RawDetection& a, const RawDetection& b) {
    // Intersection rectangle.
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    // std::max(0, ...) handles non-overlapping boxes (negative intersection).
    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection_area;

    // Guard: degenerate or non-overlapping boxes.
    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

std::vector<RawDetection> ApplyNMS(std::vector<RawDetection> detections,
                                   float iou_threshold) {
    std::vector<RawDetection> results;
    if (detections.empty()) return results;

    // ── 1. Sort by score descending — greedy NMS processes highest first ──
    std::sort(detections.begin(), detections.end(),
              [](const RawDetection& a, const RawDetection& b) {
                  return a.score > b.score;
              });

    // ── 2. Iterative suppression ──
    // keep[i] tracks whether detection i survives (not suppressed).
    std::vector<bool> keep(detections.size(), true);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (!keep[i]) continue;   // Already suppressed by an earlier detection.
        results.push_back(detections[i]);

        // Suppress lower-scoring detections of the same class.
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!keep[j]) continue;
            // Class-aware: only suppress same-class detections.
            // Different classes (safety_hat vs no_safety_hat) may overlap legitimately.
            if (detections[i].class_id == detections[j].class_id) {
                float iou = CalculateIoU(detections[i], detections[j]);
                if (iou > iou_threshold) {
                    keep[j] = false;
                }
            }
        }
    }

    return results;
}

} // namespace safety_helmet
