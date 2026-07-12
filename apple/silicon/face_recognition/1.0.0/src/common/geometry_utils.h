/**
 * @file geometry_utils.h
 * @brief 几何计算工具函数 / Geometry utility functions
 *
 * 提供边界框相关的通用几何计算（如 IoU），
 * 避免在多模块中重复实现相同逻辑。
 * Provides common geometry computations (e.g. IoU) to eliminate
 * duplicate implementations across modules.
 */
#ifndef FACE_RECOGNITION_GEOMETRY_UTILS_H
#define FACE_RECOGNITION_GEOMETRY_UTILS_H

#include "common/types.h"
#include <algorithm>

namespace face_rec {

/**
 * @brief 计算两个矩形框的交并比（IoU）
 *        Compute Intersection over Union (IoU) between two rectangles
 * @param box1 矩形框 1 / box 1
 * @param box2 矩形框 2 / box 2
 * @return IoU 值（[0, 1] 范围），无重叠时返回 0 / IoU value in [0, 1], 0 if no overlap
 */
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

} // namespace face_rec

#endif // FACE_RECOGNITION_GEOMETRY_UTILS_H
