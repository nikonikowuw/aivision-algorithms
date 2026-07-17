/**
 * @file nms.h
 * @brief Class-aware Non-Maximum Suppression (NMS).
 *
 * NMS removes redundant detections that overlap heavily with a higher-scoring
 * detection of the same class. This is class-aware: detections of different
 * classes are never suppressed against each other.
 */

#ifndef SAFETY_HELMET_NMS_H
#define SAFETY_HELMET_NMS_H

#include "detection_result.h"
#include <vector>

namespace safety_helmet {

/**
 * @brief Apply class-aware Non-Maximum Suppression.
 *
 * Algorithm (greedy):
 *   1. Sort detections by score descending.
 *   2. For each remaining detection (highest first), keep it.
 *   3. Suppress all lower-scoring detections of the SAME class
 *      whose IoU with the kept detection exceeds iou_threshold.
 *   4. Repeat until no detections remain.
 *
 * @param detections    Raw detections from the decoder (in any order).
 * @param iou_threshold  Intersection-over-Union threshold for suppression.
 *                       Typical value: 0.45.
 * @return Filtered detections, score-sorted descending.
 */
std::vector<RawDetection> ApplyNMS(std::vector<RawDetection> detections,
                                   float iou_threshold);

} // namespace safety_helmet

#endif // SAFETY_HELMET_NMS_H
