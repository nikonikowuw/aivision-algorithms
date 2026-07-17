/**
 * @file geometry_utils.h
 * @brief GPU Face Recognition — Geometry and IoU utility functions.
 * @module Utility Layer
 */

#ifndef GPU_FACE_RECOGNITION_GEOMETRY_UTILS_H
#define GPU_FACE_RECOGNITION_GEOMETRY_UTILS_H

#include "types.h"

namespace face_rec {

/**
 * @brief Compute Intersection over Union (IoU) of two bounding rectangles.
 */
inline float IoU(const Rect& a, const Rect& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float union_area = area_a + area_b - intersection;

    return (union_area > 0.0f) ? (intersection / union_area) : 0.0f;
}

/**
 * @brief Clamp a rectangle to image boundaries.
 */
inline Rect ClampRect(const Rect& r, int img_w, int img_h) {
    Rect clamped;
    clamped.x = std::max(0.0f, std::min(r.x, static_cast<float>(img_w)));
    clamped.y = std::max(0.0f, std::min(r.y, static_cast<float>(img_h)));
    clamped.width = std::min(r.width, static_cast<float>(img_w) - clamped.x);
    clamped.height = std::min(r.height, static_cast<float>(img_h) - clamped.y);
    return clamped;
}

/**
 * @brief Check if a rectangle is valid (positive width and height).
 */
inline bool IsValidRect(const Rect& r) {
    return r.width > 0.0f && r.height > 0.0f;
}

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_GEOMETRY_UTILS_H
