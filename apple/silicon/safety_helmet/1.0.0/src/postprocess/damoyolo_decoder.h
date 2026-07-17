#ifndef SAFETY_HELMET_DAMOYOLO_DECODER_H
#define SAFETY_HELMET_DAMOYOLO_DECODER_H

#include "detection_result.h"
#include <vector>

namespace safety_helmet {

bool DecodeOutputs(const std::vector<std::vector<float>>& output_tensors,
                   const std::vector<std::vector<int64_t>>& output_shapes,
                   float conf_threshold,
                   std::vector<RawDetection>& raw_detections);

} // namespace safety_helmet

#endif // SAFETY_HELMET_DAMOYOLO_DECODER_H
