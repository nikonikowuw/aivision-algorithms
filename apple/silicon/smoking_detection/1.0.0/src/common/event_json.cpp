// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Event JSON contract implementation.

#include "event_json.h"

#include <algorithm>
#include <cmath>

#include "picojson.h"

namespace smoking {
namespace {

picojson::value RectValue(const NormalizedRect& rect) {
    picojson::array values;
    values.emplace_back(static_cast<double>(rect.x));
    values.emplace_back(static_cast<double>(rect.y));
    values.emplace_back(static_cast<double>(rect.width));
    values.emplace_back(static_cast<double>(rect.height));
    return picojson::value(values);
}

}  // namespace

ErrorCode SerializeEvents(const std::vector<SmokingEvent>& events,
                          int32_t frame_width, int32_t frame_height,
                          std::string& output, std::string& error) {
    output.clear();
    error.clear();
    if (frame_width <= 0 || frame_height <= 0) {
        error = "invalid original frame dimensions";
        return ErrorCode::kSerializationFailed;
    }

    picojson::array root;
    root.reserve(events.size());
    for (const auto& event : events) {
        NormalizedRect person;
        NormalizedRect cigarette;
        if (event.track_id < 0 || event.event_id.empty() ||
            !std::isfinite(event.detect_confidence) ||
            !NormalizeRect(event.person_bbox, frame_width, frame_height, person) ||
            !NormalizeRect(event.cigarette_bbox, frame_width, frame_height, cigarette)) {
            error = "event contains invalid identifiers, confidence, or geometry";
            return ErrorCode::kSerializationFailed;
        }

        picojson::object object;
        object["category_code"] = picojson::value(static_cast<double>(event.category_code));
        object["detect_confidence"] = picojson::value(static_cast<double>(
            std::clamp(event.detect_confidence, 0.0f, 1.0f)));
        object["bbox"] = RectValue(person);
        object["cigarette_bbox"] = RectValue(cigarette);
        object["track_id"] = picojson::value(static_cast<double>(event.track_id));
        object["event_id"] = picojson::value(event.event_id);
        root.emplace_back(object);
    }

    output = picojson::value(root).serialize();
    return ErrorCode::kSuccess;
}

}  // namespace smoking
