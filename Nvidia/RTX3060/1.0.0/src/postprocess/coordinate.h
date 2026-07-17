/**
 * @file coordinate.h
 * @brief GPU Face Recognition — Coordinate conversion utilities.
 *        Converts between absolute pixel coords and normalized [0,1] coords.
 * @module Postprocessing Layer
 */

#ifndef GPU_FACE_RECOGNITION_COORDINATE_H
#define GPU_FACE_RECOGNITION_COORDINATE_H

#include "common/types.h"

namespace face_rec {

/**
 * @brief Convert absolute pixel bounding box to normalized [0,1] coordinates.
 * @param bbox Absolute pixel bbox
 * @param img_w Image width
 * @param img_h Image height
 * @return Normalized bbox [x, y, w, h] in [0,1]
 */
inline Rect NormalizeBbox(const Rect& bbox, int img_w, int img_h) {
    return {
        bbox.x / img_w,
        bbox.y / img_h,
        bbox.width / img_w,
        bbox.height / img_h
    };
}

/**
 * @brief Convert normalized [0,1] bbox back to absolute pixel coordinates.
 */
inline Rect DenormalizeBbox(const Rect& norm_bbox, int img_w, int img_h) {
    return {
        norm_bbox.x * img_w,
        norm_bbox.y * img_h,
        norm_bbox.width * img_w,
        norm_bbox.height * img_h
    };
}

/**
 * @brief Normalize an array of 5 landmarks to [0,1] coordinate space.
 */
inline std::array<Point, 5> NormalizeLandmarks(const std::array<Point, 5>& landmarks,
                                               int img_w, int img_h) {
    std::array<Point, 5> result;
    for (int i = 0; i < 5; ++i) {
        result[i] = {landmarks[i].x / img_w, landmarks[i].y / img_h};
    }
    return result;
}

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_COORDINATE_H
