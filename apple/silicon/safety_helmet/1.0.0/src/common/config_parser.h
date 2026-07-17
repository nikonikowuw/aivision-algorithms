#ifndef SAFETY_HELMET_CONFIG_PARSER_H
#define SAFETY_HELMET_CONFIG_PARSER_H

#include <string>

namespace safety_helmet {

struct AlgoConfig {
    std::string package_dir;
    std::string model_path = "weights/damoyolo_safety_helmet.onnx";
    float conf_threshold = 0.5f;
    float iou_threshold = 0.45f;

    static AlgoConfig LoadConfig(const char* config_json);
};

// Helpers
std::string DirName(const std::string& path);
std::string JoinPath(const std::string& base, const std::string& path);
bool FileExists(const std::string& path);

} // namespace safety_helmet

#endif // SAFETY_HELMET_CONFIG_PARSER_H
