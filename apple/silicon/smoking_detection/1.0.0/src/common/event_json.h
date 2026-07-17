// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Event JSON contract.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types.h"

namespace smoking {

/**
 * @brief Serialize smoking events as the ABI top-level JSON array.
 * @param events Events expressed in original-frame pixel coordinates.
 * @param frame_width Original input frame width.
 * @param frame_height Original input frame height.
 * @param output Serialized JSON on success.
 * @param error Diagnostic text on failure.
 * @return kSuccess or kSerializationFailed.
 */
ErrorCode SerializeEvents(const std::vector<SmokingEvent>& events,
                          int32_t frame_width, int32_t frame_height,
                          std::string& output, std::string& error);

}  // namespace smoking
