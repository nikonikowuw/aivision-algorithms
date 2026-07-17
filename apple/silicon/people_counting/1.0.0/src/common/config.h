/**
 * @file config.h
 * @brief People Counting configuration management
 */

#ifndef PEOPLE_COUNTING_CONFIG_H
#define PEOPLE_COUNTING_CONFIG_H

#include <string>
#include <unordered_map>
#include <vector>
#include "common/types.h"

namespace people_count {

struct CountingLine {
    std::string id;
    Point start;
    Point end;
    std::string in_direction; // "up", "down", "left", "right"
};

struct Config {
    std::string package_dir;
    std::string model_path = "weights/yolov8n.mlpackage";

    bool enable_tracker = true;
    float conf_threshold = 0.5f;
    float iou_threshold = 0.45f;
    int track_buffer = 30;
    float match_threshold = 0.2f;

    std::unordered_map<std::string, std::string> log_levels;

    bool enable_timing_report = true;
    int log_timing_interval = 100;

    static Config LoadConfig(const char* config_json);
};

// Path and file helpers
std::string DirName(const std::string& path);
std::string JoinPath(const std::string& base, const std::string& path);
bool FileExists(const std::string& path);

// Parses counting lines from context_json
std::vector<CountingLine> ParseCountingLines(const std::string& context_json);

} // namespace people_count

#endif // PEOPLE_COUNTING_CONFIG_H
