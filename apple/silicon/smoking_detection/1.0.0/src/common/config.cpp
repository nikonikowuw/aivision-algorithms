// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Configuration Parser
// PicoJSON owns JSON syntax handling; this file owns the algorithm schema.

#include "config.h"

#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>

#include "picojson.h"

namespace smoking {
namespace {

using JsonObject = picojson::object;

const std::set<std::string> kKnownKeys = {
    "backend", "analysis_fps", "max_persons", "person_conf_threshold",
    "person_nms_threshold", "cigarette_conf_threshold",
    "cigarette_nms_threshold", "tile_enabled", "tile_grid_rows",
    "tile_grid_cols", "tile_overlap", "tile_budget_per_tick",
    "upper_body_ratio", "roi_expand_x", "roi_expand_top",
    "temporal_window", "confirm_hits", "rearm_ms",
    "tracker_iou_threshold", "tracker_max_lost_ms", "detection_regions",
    "model_dir", "log_level",
};

bool ReadString(const JsonObject& object, const char* key, std::string& target,
                std::string& error) {
    const auto it = object.find(key);
    if (it == object.end()) return true;
    if (!it->second.is<std::string>()) {
        error = std::string(key) + " must be a string";
        return false;
    }
    target = it->second.get<std::string>();
    return true;
}

bool ReadBool(const JsonObject& object, const char* key, bool& target,
              std::string& error) {
    const auto it = object.find(key);
    if (it == object.end()) return true;
    if (!it->second.is<bool>()) {
        error = std::string(key) + " must be a boolean";
        return false;
    }
    target = it->second.get<bool>();
    return true;
}

bool ReadNumber(const JsonObject& object, const char* key, float& target,
                std::string& error) {
    const auto it = object.find(key);
    if (it == object.end()) return true;
    if (!it->second.is<double>() || !std::isfinite(it->second.get<double>())) {
        error = std::string(key) + " must be a finite number";
        return false;
    }
    target = static_cast<float>(it->second.get<double>());
    if (!std::isfinite(target)) {
        error = std::string(key) + " is outside the supported numeric range";
        return false;
    }
    return true;
}

template <typename Integer>
bool ReadInteger(const JsonObject& object, const char* key, Integer& target,
                 std::string& error) {
    const auto it = object.find(key);
    if (it == object.end()) return true;
    if (!it->second.is<double>()) {
        error = std::string(key) + " must be an integer";
        return false;
    }
    const double value = it->second.get<double>();
    if (!std::isfinite(value) || std::floor(value) != value ||
        value < static_cast<double>(std::numeric_limits<Integer>::min()) ||
        value > static_cast<double>(std::numeric_limits<Integer>::max())) {
        error = std::string(key) + " must be an in-range integer";
        return false;
    }
    target = static_cast<Integer>(value);
    return true;
}

double PolygonArea(const DetectionRegion& region) {
    double twice_area = 0.0;
    for (size_t i = 0, j = region.points.size() - 1;
         i < region.points.size(); j = i++) {
        twice_area += static_cast<double>(region.points[j].x) * region.points[i].y -
                      static_cast<double>(region.points[i].x) * region.points[j].y;
    }
    return std::abs(twice_area) * 0.5;
}

bool ReadDetectionRegions(const JsonObject& object,
                          std::vector<DetectionRegion>& target,
                          std::string& error) {
    const auto it = object.find("detection_regions");
    if (it == object.end()) return true;
    if (!it->second.is<picojson::array>()) {
        error = "detection_regions must be an array";
        return false;
    }

    target.clear();
    for (const auto& polygon_value : it->second.get<picojson::array>()) {
        if (!polygon_value.is<picojson::array>()) {
            error = "each detection region must be an array of [x, y] points";
            return false;
        }
        DetectionRegion region;
        for (const auto& point_value : polygon_value.get<picojson::array>()) {
            if (!point_value.is<picojson::array>()) {
                error = "detection region points must be [x, y] arrays";
                return false;
            }
            const auto& point = point_value.get<picojson::array>();
            if (point.size() != 2 || !point[0].is<double>() ||
                !point[1].is<double>()) {
                error = "detection region points must contain two numbers";
                return false;
            }
            const double x = point[0].get<double>();
            const double y = point[1].get<double>();
            if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0 || x > 1.0 ||
                y < 0.0 || y > 1.0) {
                error = "detection region coordinates must be finite and in [0, 1]";
                return false;
            }
            region.points.push_back(
                {static_cast<float>(x), static_cast<float>(y)});
        }
        if (region.points.size() < 3 || PolygonArea(region) <= 1.0e-8) {
            error = "detection regions must contain at least three non-collinear points";
            return false;
        }
        target.push_back(std::move(region));
    }
    return true;
}

bool InUnitRange(float value) {
    return std::isfinite(value) && value >= 0.0f && value <= 1.0f;
}

}  // namespace

ErrorCode ParseConfig(const char* config_json, AlgoConfig& config,
                      std::string& error_msg) {
    error_msg.clear();
    if (!config_json || *config_json == '\0') {
        return ValidateConfig(config, error_msg);
    }

    picojson::value root;
    const std::string parse_error = picojson::parse(root, config_json);
    if (!parse_error.empty() || !root.is<JsonObject>()) {
        error_msg = parse_error.empty() ? "config must be a JSON object" : parse_error;
        return ErrorCode::kInvalidParam;
    }

    const auto& object = root.get<JsonObject>();
    for (const auto& entry : object) {
        if (kKnownKeys.count(entry.first) == 0) {
            error_msg = "unknown config key: " + entry.first;
            return ErrorCode::kInvalidParam;
        }
    }

    if (!ReadString(object, "backend", config.backend, error_msg) ||
        !ReadNumber(object, "analysis_fps", config.analysis_fps, error_msg) ||
        !ReadInteger(object, "max_persons", config.max_persons, error_msg) ||
        !ReadNumber(object, "person_conf_threshold", config.person_conf_threshold, error_msg) ||
        !ReadNumber(object, "person_nms_threshold", config.person_nms_threshold, error_msg) ||
        !ReadNumber(object, "cigarette_conf_threshold", config.cigarette_conf_threshold, error_msg) ||
        !ReadNumber(object, "cigarette_nms_threshold", config.cigarette_nms_threshold, error_msg) ||
        !ReadBool(object, "tile_enabled", config.tile_enabled, error_msg) ||
        !ReadInteger(object, "tile_grid_rows", config.tile_grid_rows, error_msg) ||
        !ReadInteger(object, "tile_grid_cols", config.tile_grid_cols, error_msg) ||
        !ReadNumber(object, "tile_overlap", config.tile_overlap, error_msg) ||
        !ReadInteger(object, "tile_budget_per_tick", config.tile_budget_per_tick, error_msg) ||
        !ReadNumber(object, "upper_body_ratio", config.upper_body_ratio, error_msg) ||
        !ReadNumber(object, "roi_expand_x", config.roi_expand_x, error_msg) ||
        !ReadNumber(object, "roi_expand_top", config.roi_expand_top, error_msg) ||
        !ReadInteger(object, "temporal_window", config.temporal_window, error_msg) ||
        !ReadInteger(object, "confirm_hits", config.confirm_hits, error_msg) ||
        !ReadInteger(object, "rearm_ms", config.rearm_ms, error_msg) ||
        !ReadNumber(object, "tracker_iou_threshold", config.tracker_iou_threshold, error_msg) ||
        !ReadInteger(object, "tracker_max_lost_ms", config.tracker_max_lost_ms, error_msg) ||
        !ReadString(object, "model_dir", config.model_dir, error_msg) ||
        !ReadInteger(object, "log_level", config.log_level, error_msg) ||
        !ReadDetectionRegions(object, config.detection_regions, error_msg)) {
        return ErrorCode::kInvalidParam;
    }
    return ValidateConfig(config, error_msg);
}

ErrorCode ValidateConfig(const AlgoConfig& config, std::string& error_msg) {
    error_msg.clear();
    if (!std::isfinite(config.analysis_fps) || config.analysis_fps < 1.0f ||
        config.analysis_fps > 30.0f) {
        error_msg = "analysis_fps must be in [1, 30]";
    } else if (config.max_persons < 1 || config.max_persons > 50) {
        error_msg = "max_persons must be in [1, 50]";
    } else if (!InUnitRange(config.person_conf_threshold) ||
               !InUnitRange(config.person_nms_threshold) ||
               !InUnitRange(config.cigarette_conf_threshold) ||
               !InUnitRange(config.cigarette_nms_threshold)) {
        error_msg = "confidence and NMS thresholds must be in [0, 1]";
    } else if (config.tile_grid_rows < 1 || config.tile_grid_cols < 1 ||
               config.tile_grid_rows > 8 || config.tile_grid_cols > 8) {
        error_msg = "tile grid dimensions must be in [1, 8]";
    } else if (!std::isfinite(config.tile_overlap) || config.tile_overlap < 0.0f ||
               config.tile_overlap >= 0.5f) {
        error_msg = "tile_overlap must be in [0, 0.5)";
    } else if (config.tile_budget_per_tick < 1 ||
               config.tile_budget_per_tick > config.tile_grid_rows * config.tile_grid_cols) {
        error_msg = "tile_budget_per_tick exceeds the configured tile grid";
    } else if (!std::isfinite(config.upper_body_ratio) ||
               config.upper_body_ratio < 0.1f || config.upper_body_ratio > 1.0f ||
               !std::isfinite(config.roi_expand_x) || config.roi_expand_x < 0.0f ||
               config.roi_expand_x > 0.5f || !std::isfinite(config.roi_expand_top) ||
               config.roi_expand_top < 0.0f || config.roi_expand_top > 0.3f) {
        error_msg = "ROI ratios are outside their supported ranges";
    } else if (config.temporal_window < 3 || config.temporal_window > 20 ||
               config.confirm_hits < 1 || config.confirm_hits > config.temporal_window) {
        error_msg = "invalid temporal_window/confirm_hits combination";
    } else if (config.rearm_ms < 500 || config.rearm_ms > 10000 ||
               config.tracker_max_lost_ms < 500 || config.tracker_max_lost_ms > 5000) {
        error_msg = "rearm_ms or tracker_max_lost_ms is outside its supported range";
    } else if (!InUnitRange(config.tracker_iou_threshold)) {
        error_msg = "tracker_iou_threshold must be in [0, 1]";
    } else if (config.backend != "coreml_native") {
        error_msg = "only coreml_native is supported by this release";
    } else if (config.model_dir.empty()) {
        error_msg = "model_dir cannot be empty";
    } else if (config.log_level < 0 || config.log_level > 2) {
        error_msg = "log_level must be in [0, 2]";
    }

    return error_msg.empty() ? ErrorCode::kSuccess : ErrorCode::kInvalidParam;
}

}  // namespace smoking
