#include "config_parser.h"
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace safety_helmet {

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

static void ParseEnvFile(const std::string& path, AlgoConfig& cfg) {
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
            if (key == "conf_threshold") cfg.conf_threshold = std::stof(value);
            else if (key == "iou_threshold") cfg.iou_threshold = std::stof(value);
            else if (key == "model_path") cfg.model_path = value;
        } catch (...) {}
    }
}

AlgoConfig AlgoConfig::LoadConfig(const char* config_json) {
    AlgoConfig cfg;
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

    ExtractString(json, "model_path", &cfg.model_path);

    // Make model_path absolute if it is relative
    if (!cfg.model_path.empty() && cfg.model_path[0] != '/') {
        cfg.model_path = JoinPath(cfg.package_dir, cfg.model_path);
    }

    return cfg;
}

} // namespace safety_helmet
