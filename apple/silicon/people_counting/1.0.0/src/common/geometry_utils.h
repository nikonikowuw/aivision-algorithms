/**
 * @file geometry_utils.h
 * @brief Geometry utilities
 */

#ifndef PEOPLE_COUNTING_GEOMETRY_UTILS_H
#define PEOPLE_COUNTING_GEOMETRY_UTILS_H

#include "common/types.h"
#include <algorithm>

namespace people_count {

inline float ComputeIoU(const Rect& box1, const Rect& box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = box1.width * box1.height;
    float area2 = box2.width * box2.height;
    float union_area = area1 + area2 - intersection;

    if (union_area <= 0.0f) return 0.0f;
    return intersection / union_area;
}

} // namespace people_count

#endif // PEOPLE_COUNTING_GEOMETRY_UTILS_H
