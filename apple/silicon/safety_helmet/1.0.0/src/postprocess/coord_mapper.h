#ifndef SAFETY_HELMET_COORD_MAPPER_H
#define SAFETY_HELMET_COORD_MAPPER_H

#include "detection_result.h"
#include "preprocess/letterbox.h"
#include <vector>

namespace safety_helmet {

Detection MapToOriginal(const RawDetection& raw_det, const LetterBoxInfo& info);

} // namespace safety_helmet

#endif // SAFETY_HELMET_COORD_MAPPER_H
