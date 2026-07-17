/**
 * @file coord_mapper.h
 * @brief Coordinate mapping: letterbox space → original normalised space.
 */

#ifndef SAFETY_HELMET_COORD_MAPPER_H
#define SAFETY_HELMET_COORD_MAPPER_H

#include "detection_result.h"
#include "preprocess/letterbox.h"
#include <vector>

namespace safety_helmet {

/**
 * @brief Map a raw detection from letterbox space to normalised [0,1] space.
 *
 * Transform pipeline:
 *   1. Remove padding: (x_letterbox - pad) / ratio.
 *   2. Clip to original frame boundaries.
 *   3. Normalise: divide by original width/height → [0.0, 1.0].
 *   4. Clamp to ensure w, h ≤ 1.0 - x, y respectively.
 *
 * Also maps internal class_id (0, 1) to platform category_code:
 *   0 → 10001 ("safety_hat"),  1 → 10002 ("no_safety_hat").
 *
 * @param raw_det  Detection in letterbox 640×640 pixel coordinates.
 * @param info     Letterbox metadata (ratio, padding, original dimensions).
 * @return Detection with normalised [0,1] coordinates and category mapping.
 */
Detection MapToOriginal(const RawDetection& raw_det, const LetterBoxInfo& info);

} // namespace safety_helmet

#endif // SAFETY_HELMET_COORD_MAPPER_H
