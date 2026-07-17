/**
 * @file config.cpp
 * @brief GPU Face Recognition — Configuration loading and parsing implementation.
 */

#include "config.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cmath>

namespace face_rec {

static std::string Trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n\"");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n\"");
    return value.substr(begin, end - begin + 1);
}

static bool ExtractString(const std::string& json, const std::string& key, std::string* out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) return false;
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) return false;
    const auto quote_begin = json.find('"', colon_pos + 1);
    if (quote_begin == std::string::npos) return false;
    const auto quote_end = json.find('"', quote_begin + 1);
    if (quote_end == std::string::npos) return false;
    *out = json.substr(quote_begin + 1, quote_end - quote_begin - 1);
    return true;
}

static bool ExtractNumber(const std::string& json, const std::string& key, double* out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) return false;
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) return false;
    const auto value_begin = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_begin == std::string::npos) return false;
    const auto value_end = json.find_first_of(",}\r\n", value_begin);
    const std::string raw = json.substr(
        value_begin, value_end == std::string::npos ? std::string::npos : value_end - value_begin);
    char* end_ptr = nullptr;
    *out = std::strtod(Trim(raw).c_str(), &end_ptr);
    return true;
}

static bool ExtractBool(const std::string& json, const std::string& key, bool* out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) return false;
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) return false;
    const auto value_begin = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_begin == std::string::npos) return false;
    if (json.compare(value_begin, 4, "true") == 0) { *out = true; return true; }
    if (json.compare(value_begin, 5, "false") == 0) { *out = false; return true; }
    return false;
}

std::string DirName(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

std::string JoinPath(const std::string& base, const std::string& path) {
    if (path.empty() || path[0] == '/') return path;
    if (base.empty() || base == ".") return path;
    return base.back() == '/' ? base + path : base + "/" + path;
}

bool FileExists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

static std::string ResolveSharedLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&ResolveSharedLibraryDir), &info) != 0 && info.dli_fname) {
        return DirName(info.dli_fname);
    }
    return ".";
}

static void ParseEnvFile(const std::string& path, Config& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) return;
    std::string line;
    while (std::getline(file, line)) {
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        auto trim_begin = line.find_first_not_of(" \t\r\n");
        if (trim_begin == std::string::npos) continue;
        auto trim_end = line.find_last_not_of(" \t\r\n");
        line = line.substr(trim_begin, trim_end - trim_begin + 1);
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        auto k_begin = key.find_first_not_of(" \t\r\n\"");
        auto k_end = key.find_last_not_of(" \t\r\n\"");
        if (k_begin != std::string::npos) key = key.substr(k_begin, k_end - k_begin + 1);
        auto v_begin = value.find_first_not_of(" \t\r\n\"");
        auto v_end = value.find_last_not_of(" \t\r\n\"");
        if (v_begin != std::string::npos) value = value.substr(v_begin, v_end - v_begin + 1);

        try {
            if (key == "enable_tracker") cfg.enable_tracker = (value == "true");
            else if (key == "person_conf_thres") cfg.person_conf_thres = std::stof(value);
            else if (key == "face_conf_thres") cfg.face_conf_thres = std::stof(value);
            else if (key == "recognition_threshold") cfg.recognition_threshold = std::stof(value);
            else if (key == "nms_iou_thres") cfg.nms_iou_thres = std::stof(value);
            else if (key == "tracker_iou_thres") cfg.tracker_iou_thres = std::stof(value);
            else if (key == "tracker_max_lost_frames") cfg.tracker_max_lost_frames = std::stoi(value);
            else if (key == "max_batch_size") cfg.max_batch_size = std::stoi(value);
            else if (key == "batch_timeout_ms") cfg.batch_timeout_ms = std::stoi(value);
            else if (key == "max_persons") cfg.max_persons = std::stoi(value);
            else if (key == "max_faces") cfg.max_faces = std::stoi(value);
            else if (key == "cuda_device_id") cfg.cuda_device_id = std::stoi(value);
            else if (key == "trt_max_workspace_size") cfg.trt_max_workspace_size = static_cast<size_t>(std::stoull(value));
            else if (key == "yolo11n_engine_path") cfg.yolo11n_engine_path = value;
            else if (key == "retinaface_engine_path") cfg.retinaface_engine_path = value;
            else if (key == "adaface_ir50_engine_path") cfg.adaface_ir50_engine_path = value;
            else if (key == "enable_timing_report") cfg.enable_timing_report = (value == "true");
            else if (key == "log_timing_interval") cfg.log_timing_interval = std::stoi(value);
            else if (key == "log_model_io") cfg.log_model_io = (value == "true");
            else if (key.rfind("log_level_", 0) == 0) {
                std::string mod_name = key.substr(10);
                cfg.log_levels[mod_name] = value;
            }
        } catch (...) {
            // Ignore parse errors in .env
        }
    }
}

Config Config::LoadConfig(const char* config_json) {
    Config cfg;
    std::string json(config_json ? config_json : "");

    // 1. Resolve package_dir
    if (!ExtractString(json, "package_dir", &cfg.package_dir) || cfg.package_dir.empty()) {
        cfg.package_dir = ResolveSharedLibraryDir();
    }

    // 2. Load .env
    std::string env_path = JoinPath(cfg.package_dir, ".env");
    ParseEnvFile(env_path, cfg);

    // 3. Overlay runtime config_json
    double value = 0.0;
    if (ExtractNumber(json, "person_conf_thres", &value))
        cfg.person_conf_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "face_conf_thres", &value))
        cfg.face_conf_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "recognition_threshold", &value))
        cfg.recognition_threshold = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "nms_iou_thres", &value))
        cfg.nms_iou_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "tracker_iou_thres", &value))
        cfg.tracker_iou_thres = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "tracker_max_lost_frames", &value))
        cfg.tracker_max_lost_frames = std::clamp(static_cast<int>(value), 1, 300);
    if (ExtractNumber(json, "max_batch_size", &value))
        cfg.max_batch_size = std::clamp(static_cast<int>(value), 1, 16);
    if (ExtractNumber(json, "batch_timeout_ms", &value))
        cfg.batch_timeout_ms = std::clamp(static_cast<int>(value), 1, 100);
    if (ExtractNumber(json, "max_persons", &value))
        cfg.max_persons = std::clamp(static_cast<int>(value), 1, 200);
    if (ExtractNumber(json, "max_faces", &value))
        cfg.max_faces = std::clamp(static_cast<int>(value), 1, 200);
    if (ExtractNumber(json, "cuda_device_id", &value))
        cfg.cuda_device_id = std::clamp(static_cast<int>(value), 0, 15);
    if (ExtractNumber(json, "log_timing_interval", &value))
        cfg.log_timing_interval = std::clamp(static_cast<int>(value), 1, 10000);

    // Boolean flags
    ExtractBool(json, "enable_tracker", &cfg.enable_tracker);
    ExtractBool(json, "enable_timing_report", &cfg.enable_timing_report);
    ExtractBool(json, "log_model_io", &cfg.log_model_io);

    // Override model paths
    ExtractString(json, "yolo11n_engine_path", &cfg.yolo11n_engine_path);
    ExtractString(json, "retinaface_engine_path", &cfg.retinaface_engine_path);
    ExtractString(json, "adaface_ir50_engine_path", &cfg.adaface_ir50_engine_path);

    // Parse per-module log levels from JSON
    std::string log_val;
    std::vector<std::string> modules = {
        "config", "backend", "person_det", "face_det", "face_rec",
        "tracker", "face_index", "aligner", "image_util", "pipeline",
        "app", "body_attr", "face_attr"
    };
    for (const auto& mod : modules) {
        if (ExtractString(json, "log_level_" + mod, &log_val)) {
            cfg.log_levels[mod] = log_val;
        }
    }

    // 4. Initialize Logger
    Logger::Instance().Initialize(cfg.log_levels);

    return cfg;
}

} // namespace face_rec
