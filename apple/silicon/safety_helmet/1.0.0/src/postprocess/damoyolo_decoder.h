/**
 * @file damoyolo_decoder.h
 * @brief DAMO-YOLO-S output decoder — converts raw tensor to RawDetection list.
 */

#ifndef SAFETY_HELMET_DAMOYOLO_DECODER_H
#define SAFETY_HELMET_DAMOYOLO_DECODER_H

#include "detection_result.h"
#include <vector>

namespace safety_helmet {

/**
 * @brief Decode DAMO-YOLO-S output tensor into scored detections.
 *
 * DAMO-YOLO-S outputs a tensor of shape [1, N, 6] where each anchor row is:
 *   [x1, y1, x2, y2, score_class0, score_class1]
 *
 * Unlike standard YOLO, DAMO-YOLO uses a per-anchor dual-score format
 * instead of a class-probability vector. For each anchor, the maximum
 * score determines the class, and only anchors exceeding conf_threshold
 * are retained.
 *
 * Coordinates (x1, y1, x2, y2) are in letterbox 640×640 pixel space.
 *
 * @param output_tensors   Raw float tensors from ONNXRuntime (expects exactly 1 tensor).
 * @param output_shapes    Corresponding tensor shapes (expects [1, N, 6]).
 * @param conf_threshold   Minimum confidence to retain a detection.
 * @param raw_detections   [out] Filtered list of RawDetection in letterbox space.
 * @return true on success, false if tensor format is unexpected.
 */
bool DecodeOutputs(const std::vector<std::vector<float>>& output_tensors,
                   const std::vector<std::vector<int64_t>>& output_shapes,
                   float conf_threshold,
                   std::vector<RawDetection>& raw_detections);

} // namespace safety_helmet

#endif // SAFETY_HELMET_DAMOYOLO_DECODER_H
