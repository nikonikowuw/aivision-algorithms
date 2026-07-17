/**
 * @file config_parser.cpp
 * @brief Configuration loading — manual JSON parser + .env overlay.
 *
 * ## Design rationale
 *
 * The algorithm .so must be self-contained: no external JSON library
 * dependency. This file implements a minimal flat-JSON key-value extractor
 * sufficient for the simple config objects we receive from the Engine.
 * It is NOT a general-purpose JSON parser — it assumes a flat object
 * with string/number values and no nesting, arrays, or escapes beyond
 * basic double-quoted strings.
 *
 * ## 3-layer config resolution
 *
 *   Layer 1 — AlgoConfig default member initialisers (compile-time defaults).
 *   Layer 2 — .env file in package_dir (deployment-specific defaults).
 *   Layer 3 — config_json from detector_init (runtime overrides, highest priority).
 *
 * Each layer overwrites values from previous layers. Values are clamped
 * to [0.0, 1.0] at Layer 3 to prevent invalid thresholds.
 */

#include "config_parser.h"
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace safety_helmet {

// ---------------------------------------------------------------------------
// Utility: trim whitespace and surrounding quotes from a string value.
// ---------------------------------------------------------------------------
static std::string Trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n\"");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n\"");
    return value.substr(begin, end - begin + 1);
}

// ---------------------------------------------------------------------------
// ExtractString — find "key": "..." and return the quoted value.
//
// Assumes flat JSON: no nesting, no escaped quotes inside values.
// Returns false if the key is not found.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// ExtractNumber — find "key": <number> and parse as double.
//
// Handles integer and floating-point literals (no scientific notation).
// Returns false if key not found or parsing fails.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Filesystem helpers.
// ---------------------------------------------------------------------------

std::string DirName(const std::string& path) {
    const auto last_slash = path.find_last_of('/');
    if (last_slash == std::string::npos) return ".";
    if (last_slash == 0) return "/";   // Root directory edge case.
    return path.substr(0, last_slash);
}

std::string JoinPath(const std::string& base, const std::string& path) {
    if (base.empty() || base == ".") return path;
    if (path.empty()) return base;
    // Avoid double separator.
    if (base.back() == '/' || path.front() == '/') return base + path;
    return base + "/" + path;
}

bool FileExists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

// ---------------------------------------------------------------------------
// ResolveSharedLibraryDir — find the directory containing this .so at runtime.
//
// Uses dladdr(self) to discover the shared library's filesystem path,
// then extracts the directory component. This is more robust than relying
// on argv[0] or $PWD.
// ---------------------------------------------------------------------------
static std::string ResolveSharedLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&ResolveSharedLibraryDir), &info) != 0 && info.dli_fname) {
        return DirName(info.dli_fname);
    }
    return ".";
}

// ---------------------------------------------------------------------------
// ParseEnvFile — read key=value pairs from a .env file.
//
// Format: one KEY=VALUE per line. Lines starting with # are comments.
// Empty lines and lines without = are skipped. Only recognised keys
// are applied; unknown keys are silently ignored.
// ---------------------------------------------------------------------------
static void ParseEnvFile(const std::string& path, AlgoConfig& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) return;
    std::string line;
    while (std::getline(file, line)) {
        // Strip trailing comments (#).
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        // Trim whitespace.
        auto trim_begin = line.find_first_not_of(" \t\r\n");
        if (trim_begin == std::string::npos) continue;
        auto trim_end = line.find_last_not_of(" \t\r\n");
        line = line.substr(trim_begin, trim_end - trim_begin + 1);

        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Trim quotes and whitespace from key and value.
        auto k_begin = key.find_first_not_of(" \t\r\n\"");
        auto k_end = key.find_last_not_of(" \t\r\n\"");
        if (k_begin != std::string::npos) key = key.substr(k_begin, k_end - k_begin + 1);
        
        auto v_begin = value.find_first_not_of(" \t\r\n\"");
        auto v_end = value.find_last_not_of(" \t\r\n\"");
        if (v_begin != std::string::npos) value = value.substr(v_begin, v_end - v_begin + 1);

        // Apply recognised keys. stof may throw; catch and ignore malformed values.
        try {
            if (key == "conf_threshold") cfg.conf_threshold = std::stof(value);
            else if (key == "iou_threshold") cfg.iou_threshold = std::stof(value);
            else if (key == "model_path") cfg.model_path = value;
        } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// LoadConfig — 3-layer config resolution.
// ---------------------------------------------------------------------------
AlgoConfig AlgoConfig::LoadConfig(const char* config_json) {
    AlgoConfig cfg;
    std::string json(config_json ? config_json : "");

    // ── Layer 1: defaults already set by struct initialisers ──

    // ── Layer 2: resolve package_dir and load .env ──
    // package_dir is needed to resolve relative paths for model and .env.
    if (!ExtractString(json, "package_dir", &cfg.package_dir) || cfg.package_dir.empty()) {
        cfg.package_dir = ResolveSharedLibraryDir();
    }

    // Load .env defaults from the package directory.
    std::string env_path = JoinPath(cfg.package_dir, ".env");
    ParseEnvFile(env_path, cfg);

    // ── Layer 3: overlay runtime config_json (highest priority) ──
    // Thresholds are clamped to [0.0, 1.0] to prevent invalid values from
    // reaching the inference pipeline.
    double value = 0.0;
    if (ExtractNumber(json, "conf_threshold", &value))
        cfg.conf_threshold = std::clamp(static_cast<float>(value), 0.0f, 1.0f);
    if (ExtractNumber(json, "iou_threshold", &value))
        cfg.iou_threshold = std::clamp(static_cast<float>(value), 0.0f, 1.0f);

    ExtractString(json, "model_path", &cfg.model_path);

    // Resolve relative model path against package_dir.
    if (!cfg.model_path.empty() && cfg.model_path[0] != '/') {
        cfg.model_path = JoinPath(cfg.package_dir, cfg.model_path);
    }

    return cfg;
}

} // namespace safety_helmet
