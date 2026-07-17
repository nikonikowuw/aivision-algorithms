/**
 * @file detection_result.h
 * @brief Detection data structures for the two coordinate spaces in the pipeline.
 *
 * Two structs represent the same detection at different stages:
 *
 *   RawDetection  — Model output space (letterbox 640×640 coordinates).
 *                   Produced by DecodeOutputs, consumed by ApplyNMS.
 *   Detection     — Final normalised output ([0,1] relative to original frame).
 *                   Produced by MapToOriginal, consumed by SerializeDetections.
 */

#ifndef SAFETY_HELMET_DETECTION_RESULT_H
#define SAFETY_HELMET_DETECTION_RESULT_H

#include <string>

namespace safety_helmet {

/**
 * @brief Normalised detection result for C-ABI output.
 *
 * All coordinate fields are in [0.0, 1.0] relative to the original,
 * unpreprocessed frame dimensions. x, y are the top-left corner;
 * w, h are width and height.
 */
struct Detection {
    float x, y, w, h;           ///< Bounding box in normalised [0,1] coordinates
    float confidence;            ///< Detection confidence score [0,1]
    int category_code;           ///< Platform category code (10001=safety_hat, 10002=no_safety_hat)
    std::string category_name;   ///< Human-readable label
};

/**
 * @brief Raw detection in letterbox model-output space.
 *
 * x1, y1, x2, y2 are in 640×640 letterbox pixel coordinates.
 * class_id is 0 (safety_hat) or 1 (no_safety_hat) — the internal
 * model class index before mapping to platform category_code.
 */
struct RawDetection {
    float x1, y1, x2, y2;   ///< Bounding box corners in letterbox pixel space
    float score;             ///< Detection confidence score [0,1]
    int class_id;            ///< Internal model class index (0 or 1)
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_DETECTION_RESULT_H
