/**
 * @file config.cpp
 * @brief Configuration loading and parsing implementation
 */

#include "config.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>

namespace people_count {

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
    std::string val = Trim(json.substr(value_begin, 5));
    if (val.rfind("true", 0) == 0) {
        *out = true;
        return true;
    } else if (val.rfind("false", 0) == 0) {
        *out = false;
        return true;
    }
    return false;
}

std::string DirName(const std::string& path) {
    const auto last_slash = path.find_last_of('/');
    if (last_slash == std::string::npos) return ".";
    if (last_slash == 0) return "/";
    return path.substr(0, last_slash);
}

std::string JoinPath(const std::string& base, const std::string& path) {
    if (base.empty() || base == ".") return path;
    if (path.empty()) return base;
    if (base.back() == '/' || path.front() == '/') return base + path;
    return base + "/" + path;
}

bool FileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static std::string ResolveSharedLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&ResolveSharedLibraryDir), &info) != 0 && info.dli_fname) {
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
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
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
            if (key == "enable_tracker") cfg.enable_tracker = (value == "true" || value == "1");
            else if (key == "conf_threshold") cfg.conf_threshold = std::stof(value);
            else if (key == "iou_threshold") cfg.iou_threshold = std::stof(value);
            else if (key == "track_buffer") cfg.track_buffer = std::stoi(value);
            else if (key == "match_threshold") cfg.match_threshold = std::stof(value);
            else if (key == "enable_timing_report") cfg.enable_timing_report = (value == "true" || value == "1");
            else if (key == "log_timing_interval") cfg.log_timing_interval = std::stoi(value);
            else if (key == "model_path") cfg.model_path = value;
            else if (key.rfind("log_level_", 0) == 0) {
                std::string mod_name = key.substr(10);
                cfg.log_levels[mod_name] = value;
            }
        } catch (...) {}
    }
}

Config Config::LoadConfig(const char* config_json) {
    Config cfg;
    std::string json(config_json ? config_json : "");

    // 1. Resolve package_dir
    if (!ExtractString(json, "package_dir", &cfg.package_dir) || cfg.package_dir.empty()) {
        cfg.package_dir = ResolveSharedLibraryDir();
    }

    // 2. Load .env default parameters
    std::string env_path = JoinPath(cfg.package_dir, ".env");
    ParseEnvFile(env_path, cfg);

    // 3. Overlay runtime config_json parameters
    double value = 0.0;
    if (ExtractNumber(json, "conf_threshold", &value))
        cfg.conf_threshold = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "iou_threshold", &value))
        cfg.iou_threshold = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "track_buffer", &value))
        cfg.track_buffer = std::clamp(static_cast<int>(value), 1, 300);
    if (ExtractNumber(json, "match_threshold", &value))
        cfg.match_threshold = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "log_timing_interval", &value))
        cfg.log_timing_interval = std::clamp(static_cast<int>(value), 1, 10000);

    ExtractBool(json, "enable_tracker", &cfg.enable_tracker);
    ExtractBool(json, "enable_timing_report", &cfg.enable_timing_report);
    ExtractString(json, "model_path", &cfg.model_path);

    std::vector<std::string> modules = {
        "config", "backend", "detector", "tracker", "pipeline", "app"
    };
    std::string log_val;
    for (const auto &mod : modules) {
        if (ExtractString(json, "log_level_" + mod, &log_val)) {
            cfg.log_levels[mod] = log_val;
        }
    }

    Logger::Instance().Initialize(cfg.log_levels);

    return cfg;
}

std::vector<CountingLine> ParseCountingLines(const std::string& json) {
    std::vector<CountingLine> lines;
    auto key_pos = json.find("\"counting_lines\"");
    if (key_pos == std::string::npos) return lines;

    auto start_bracket = json.find('[', key_pos + 16);
    if (start_bracket == std::string::npos) return lines;

    // Find closing bracket matching start_bracket (simplified check)
    int depth = 1;
    size_t end_bracket = start_bracket + 1;
    for (; end_bracket < json.size(); ++end_bracket) {
        if (json[end_bracket] == '[') depth++;
        else if (json[end_bracket] == ']') {
            depth--;
            if (depth == 0) break;
        }
    }
    if (depth != 0 || end_bracket >= json.size()) return lines;

    std::string array_content = json.substr(start_bracket + 1, end_bracket - start_bracket - 1);
    
    // Parse objects inside the array: { ... }
    size_t pos = 0;
    while (true) {
        auto obj_start = array_content.find('{', pos);
        if (obj_start == std::string::npos) break;
        auto obj_end = array_content.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj_str = array_content.substr(obj_start, obj_end - obj_start + 1);
        pos = obj_end + 1;

        CountingLine line;
        
        // Extract string values
        ExtractString(obj_str, "id", &line.id);
        ExtractString(obj_str, "in_direction", &line.in_direction);

        // Extract "start": [x, y]
        auto start_pos = obj_str.find("\"start\"");
        if (start_pos != std::string::npos) {
            auto s_bracket = obj_str.find('[', start_pos);
            auto e_bracket = obj_str.find(']', s_bracket);
            if (s_bracket != std::string::npos && e_bracket != std::string::npos) {
                std::string coords = obj_str.substr(s_bracket + 1, e_bracket - s_bracket - 1);
                auto comma = coords.find(',');
                if (comma != std::string::npos) {
                    char* end_ptr = nullptr;
                    line.start.x = static_cast<float>(std::strtod(coords.substr(0, comma).c_str(), &end_ptr));
                    line.start.y = static_cast<float>(std::strtod(coords.substr(comma + 1).c_str(), &end_ptr));
                }
            }
        }

        // Extract "end": [x, y]
        auto end_pos = obj_str.find("\"end\"");
        if (end_pos != std::string::npos) {
            auto s_bracket = obj_str.find('[', end_pos);
            auto e_bracket = obj_str.find(']', s_bracket);
            if (s_bracket != std::string::npos && e_bracket != std::string::npos) {
                std::string coords = obj_str.substr(s_bracket + 1, e_bracket - s_bracket - 1);
                auto comma = coords.find(',');
                if (comma != std::string::npos) {
                    char* end_ptr = nullptr;
                    line.end.x = static_cast<float>(std::strtod(coords.substr(0, comma).c_str(), &end_ptr));
                    line.end.y = static_cast<float>(std::strtod(coords.substr(comma + 1).c_str(), &end_ptr));
                }
            }
        }

        if (!line.id.empty()) {
            lines.push_back(line);
        }
    }

    return lines;
}

} // namespace people_count
