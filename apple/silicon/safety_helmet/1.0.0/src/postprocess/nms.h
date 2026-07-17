#ifndef SAFETY_HELMET_NMS_H
#define SAFETY_HELMET_NMS_H

#include "detection_result.h"
#include <vector>

namespace safety_helmet {

std::vector<RawDetection> ApplyNMS(const std::vector<RawDetection>& detections, float iou_threshold);

} // namespace safety_helmet

#endif // SAFETY_HELMET_NMS_H
