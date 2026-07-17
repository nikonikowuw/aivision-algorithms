/**
 * @file config_parser.h
 * @brief Algorithm configuration parsing — JSON + .env overlay.
 *
 * Config loading uses a 3-layer resolution strategy:
 *   1. Default values from the struct initialisers.
 *   2. .env file in the package directory (if present).
 *   3. Runtime config_json from detector_init().
 *
 * Each layer overrides the previous — runtime JSON has highest priority.
 *
 * JSON parsing is intentionally zero-dependency (no nlohmann/json) to keep
 * the .so self-contained. A minimal manual parser extracts string and number
 * values from a flat JSON object.
 */

#ifndef SAFETY_HELMET_CONFIG_PARSER_H
#define SAFETY_HELMET_CONFIG_PARSER_H

#include <string>

namespace safety_helmet {

/**
 * @brief Algorithm runtime configuration.
 *
 * Fields are populated by LoadConfig() using the 3-layer overlay strategy.
 * All thresholds are in [0.0, 1.0] and are clamped by LoadConfig().
 */
struct AlgoConfig {
    std::string package_dir;                                       ///< Directory containing this .so
    std::string model_path = "weights/damoyolo_safety_helmet.onnx"; ///< Absolute path to ONNX model
    float conf_threshold = 0.5f;                                   ///< Confidence threshold [0,1]
    float iou_threshold = 0.45f;                                   ///< NMS IoU threshold [0,1]

    /**
     * @brief Load config with 3-layer overlay: defaults → .env → runtime JSON.
     *
     * @param config_json  Flat JSON string from detector_init (may be null).
     * @return Populated AlgoConfig with values clamped to valid ranges.
     */
    static AlgoConfig LoadConfig(const char* config_json);
};

// Filesystem helpers used by detector_self_test and config resolution.

std::string DirName(const std::string& path);
std::string JoinPath(const std::string& base, const std::string& path);
bool FileExists(const std::string& path);

} // namespace safety_helmet

#endif // SAFETY_HELMET_CONFIG_PARSER_H
