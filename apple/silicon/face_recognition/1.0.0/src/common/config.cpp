/**
 * @file config.cpp
 * @brief 配置加载与解析实现
 *        Configuration loading and parsing implementation.
 * @module 基础配置层 (Config Layer)
 * @details 实现三级优先级配置加载机制，包含 JSON 键值提取工具函数、
 *          .env 文件解析、动态库路径自动解析等核心逻辑。
 *          采用手工解析而非第三方 JSON 库以减少（Vision Framework 环境下的）依赖。
 */

#include "config.h"
#include "logger.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>

namespace face_rec {

/**
 * @brief 去除字符串两端的空白字符和引号
 *        Strip whitespace and quotes from both ends of a string.
 * @param value 输入字符串
 * @return 修剪后的字符串
 */
static std::string Trim(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n\"");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n\"");
    return value.substr(begin, end - begin + 1);
}

/**
 * @brief 从 JSON 字符串中提取指定 key 的字符串值
 *        Extract a string value by key from an ad-hoc JSON string.
 * @details 轻量级 JSON 键值提取，不依赖 JSON 解析库。
 *          查找 "key": "value" 模式并提取引号内的内容。
 * @param json  JSON 格式字符串
 * @param key   目标键名
 * @param out   输出缓冲区
 * @return true 如果找到并提取成功
 */
static bool ExtractString(const std::string &json, const std::string &key, std::string *out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    const auto quote_begin = json.find('"', colon_pos + 1);
    if (quote_begin == std::string::npos) {
        return false;
    }
    const auto quote_end = json.find('"', quote_begin + 1);
    if (quote_end == std::string::npos) {
        return false;
    }
    *out = json.substr(quote_begin + 1, quote_end - quote_begin - 1);
    return true;
}

/**
 * @brief 从 JSON 字符串中提取指定 key 的数值
 *        Extract a numeric value by key from a JSON string.
 * @details 使用 strtod 解析，支持整数和浮点数格式。
 * @param json JSON 格式字符串
 * @param key  目标键名
 * @param out  输出（double 类型）
 * @return true 如果找到并解析成功
 */
static bool ExtractNumber(const std::string &json, const std::string &key, double *out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    const auto value_begin = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_begin == std::string::npos) {
        return false;
    }
    const auto value_end = json.find_first_of(",}\r\n", value_begin);
    const std::string raw = json.substr(
        value_begin, value_end == std::string::npos ? std::string::npos : value_end - value_begin);
    char *end_ptr = nullptr;
    const double parsed = std::strtod(Trim(raw).c_str(), &end_ptr);
    *out = parsed;
    return true;
}

/**
 * @brief 从 JSON 字符串中提取指定 key 的布尔值
 *        Extract a boolean value by key from a JSON string.
 * @details 匹配 "key": true 或 "key": false 模式。
 * @param json JSON 格式字符串
 * @param key  目标键名
 * @param out  输出布尔值
 * @return true 如果找到并识别为 "true" 或 "false"
 */
static bool ExtractBool(const std::string &json, const std::string &key, bool *out) {
    const std::string pattern = "\"" + key + "\"";
    const auto key_pos = json.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    const auto colon_pos = json.find(':', key_pos + pattern.size());
    if (colon_pos == std::string::npos) {
        return false;
    }
    const auto value_begin = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (value_begin == std::string::npos) {
        return false;
    }
    if (json.compare(value_begin, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (json.compare(value_begin, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

/**
 * @brief 获取路径中的目录部分
 *        Extract the parent directory from a file path.
 * @param path 文件或目录路径
 * @return 目录路径，根目录返回 "/"，无分隔符时返回 "."
 */
std::string DirName(const std::string &path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

/**
 * @brief 拼接基础路径和相对路径
 *        Join a base directory with a relative path component.
 * @details 自动处理分隔符：如果 path 以 '/' 开头则直接返回（绝对路径），
 *          否则将 base 和 path 用 '/' 连接。
 * @param base 基础目录路径
 * @param path 相对路径（如为绝对路径则直接返回）
 * @return 拼接后的完整路径
 */
std::string JoinPath(const std::string &base, const std::string &path) {
    if (path.empty() || path[0] == '/') {
        return path;
    }
    if (base.empty() || base == ".") {
        return path;
    }
    return base.back() == '/' ? base + path : base + "/" + path;
}

/**
 * @brief 检查文件是否存在
 *        Check whether a file exists on the filesystem.
 * @param path 目标文件路径
 * @return true 如果文件存在且可访问
 */
bool FileExists(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

/**
 * @brief 解析动态库所在目录路径
 *        Resolve the directory containing the currently loaded shared library.
 * @details 使用 dladdr 获取当前函数所在动态库的路径，再取其目录名。
 *          这在 macOS/iOS 环境下用于定位与 .dylib 同目录的资源文件。
 * @return 动态库所在目录路径，失败时返回 "."
 */
static std::string ResolveSharedLibraryDir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&ResolveSharedLibraryDir), &info) != 0 && info.dli_fname) {
        return DirName(info.dli_fname);
    }
    return ".";
}

/**
 * @brief 解析 .env 配置文件
 *        Parse a .env key-value configuration file.
 * @details 支持以下格式：
 *          - 注释：以 '#' 开头的行会被忽略
 *          - 键值对：KEY=VALUE，支持 " 引号包围的值
 *          - 特殊前缀：log_level_ 开头的键会写入 cfg.log_levels 映射表
 *          @note 此解析发生在 JSON 配置之前，因此 .env 设置会被 JSON 值覆盖。
 * @param path .env 文件路径
 * @param cfg  待更新的配置对象
 */
static void ParseEnvFile(const std::string &path, Config &cfg) {
    std::ifstream file(path);
    if (!file.is_open()) {
        // .env 文件不存在不是错误，使用默认值即可
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing comments
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        // Trim leading/trailing whitespace
        auto trim_begin = line.find_first_not_of(" \t\r\n");
        if (trim_begin == std::string::npos) continue;
        auto trim_end = line.find_last_not_of(" \t\r\n");
        line = line.substr(trim_begin, trim_end - trim_begin + 1);

        // Find key-value separator
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Trim key & value
        auto k_begin = key.find_first_not_of(" \t\r\n\"");
        auto k_end = key.find_last_not_of(" \t\r\n\"");
        if (k_begin != std::string::npos) key = key.substr(k_begin, k_end - k_begin + 1);
        
        auto v_begin = value.find_first_not_of(" \t\r\n\"");
        auto v_end = value.find_last_not_of(" \t\r\n\"");
        if (v_begin != std::string::npos) value = value.substr(v_begin, v_end - v_begin + 1);

        // Map key to Config field
        try {
            if (key == "enable_tracker") cfg.enable_tracker = (value == "true");
            else if (key == "person_conf_thres") cfg.person_conf_thres = std::stof(value);
            else if (key == "face_conf_thres") cfg.face_conf_thres = std::stof(value);
            else if (key == "recognition_threshold") cfg.recognition_threshold = std::stof(value);
            else if (key == "zero_copy_required") cfg.zero_copy_required = (value == "true");
            else if (key == "allow_cpu_fallback") cfg.allow_cpu_fallback = (value == "true");
            else if (key == "nms_iou_thres") cfg.nms_iou_thres = std::stof(value);
            else if (key == "tracker_iou_thres") cfg.tracker_iou_thres = std::stof(value);
            else if (key == "tracker_max_lost_frames") cfg.tracker_max_lost_frames = std::stoi(value);
            else if (key == "max_persons") cfg.max_persons = std::stoi(value);
            else if (key == "max_faces") cfg.max_faces = std::stoi(value);
            else if (key == "thread_pool_size") cfg.thread_pool_size = std::stoi(value);
            else if (key == "ort_intra_op_threads") cfg.ort_intra_op_threads = std::stoi(value);
            else if (key == "min_body_height") cfg.head_roi.min_body_height = std::stoi(value);
            else if (key == "full_body_ratio") cfg.head_roi.full_body_ratio = std::stof(value);
            else if (key == "half_body_ratio") cfg.head_roi.half_body_ratio = std::stof(value);
            else if (key == "crouch_ratio") cfg.head_roi.crouch_ratio = std::stof(value);
            else if (key == "width_expand") cfg.head_roi.width_expand = std::stof(value);
            else if (key == "enable_timing_report") cfg.enable_timing_report = (value == "true");
            else if (key == "log_timing_interval") cfg.log_timing_interval = std::stoi(value);
            else if (key == "log_model_io") cfg.log_model_io = (value == "true");
            else if (key == "scrfd_person_model_path") cfg.scrfd_person_model_path = value;
            else if (key == "scrfd_500m_model_path") cfg.scrfd_500m_model_path = value;
            else if (key == "adaface_model_path") cfg.adaface_model_path = value;
            else if (key.rfind("log_level_", 0) == 0) {
                std::string mod_name = key.substr(10);
                cfg.log_levels[mod_name] = value;
            }
        } catch (...) {
            // Ignore parse errors in .env
        }
    }
}

/**
 * @brief 加载并合并全部配置（三级优先级）
 *        Load and merge configuration from all tiers (3-tier priority).
 * @details 加载流程：
 *          1. 从 config_json 中提取 package_dir（如未指定则自动解析动态库目录）
 *          2. 加载 .env 文件覆盖编译默认值
 *          3. 加载 config_json 中的各项参数覆盖 .env 值
 *          4. 根据最终 log_levels 初始化日志模块
 * @note 所有数值型参数在从 JSON 加载时都会经过 clamp 范围校验，
 *       确保不会出现越界值导致运行时异常。
 * @param config_json 运行时 JSON 配置字符串（可为 nullptr）
 * @return 合并后的完整配置对象
 */
Config Config::LoadConfig(const char *config_json) {
    Config cfg;
    std::string json(config_json ? config_json : "");

    // 1. Resolve package_dir first (either from JSON or default to .so location)
    if (!ExtractString(json, "package_dir", &cfg.package_dir) || cfg.package_dir.empty()) {
        cfg.package_dir = ResolveSharedLibraryDir();
    }

    // 2. Load .env default parameters
    std::string env_path = JoinPath(cfg.package_dir, ".env");
    ParseEnvFile(env_path, cfg);

    // 3. Overlay runtime config_json parameters
    // Numeric thresholds are clamped to valid ranges
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
    if (ExtractNumber(json, "max_persons", &value))
        cfg.max_persons = std::clamp(static_cast<int>(value), 1, 200);
    if (ExtractNumber(json, "max_faces", &value))
        cfg.max_faces = std::clamp(static_cast<int>(value), 1, 200);
    if (ExtractNumber(json, "thread_pool_size", &value))
        cfg.thread_pool_size = std::clamp(static_cast<int>(value), 1, 64);
    if (ExtractNumber(json, "ort_intra_op_threads", &value))
        cfg.ort_intra_op_threads = std::clamp(static_cast<int>(value), 1, 64);
    if (ExtractNumber(json, "log_timing_interval", &value))
        cfg.log_timing_interval = std::clamp(static_cast<int>(value), 1, 10000);
    if (ExtractNumber(json, "min_body_height", &value))
        cfg.head_roi.min_body_height = static_cast<int>(value);
    if (ExtractNumber(json, "full_body_ratio", &value))
        cfg.head_roi.full_body_ratio = static_cast<float>(value);
    if (ExtractNumber(json, "half_body_ratio", &value))
        cfg.head_roi.half_body_ratio = static_cast<float>(value);
    if (ExtractNumber(json, "crouch_ratio", &value))
        cfg.head_roi.crouch_ratio = static_cast<float>(value);
    if (ExtractNumber(json, "width_expand", &value))
        cfg.head_roi.width_expand = static_cast<float>(value);

    // Boolean flags
    ExtractBool(json, "enable_tracker", &cfg.enable_tracker);
    ExtractBool(json, "zero_copy_required", &cfg.zero_copy_required);
    ExtractBool(json, "allow_cpu_fallback", &cfg.allow_cpu_fallback);
    ExtractBool(json, "enable_timing_report", &cfg.enable_timing_report);
    ExtractBool(json, "log_model_io", &cfg.log_model_io);

    // Override model paths from JSON if specified
    ExtractString(json, "scrfd_person_model_path", &cfg.scrfd_person_model_path);
    ExtractString(json, "scrfd_500m_model_path", &cfg.scrfd_500m_model_path);
    ExtractString(json, "adaface_model_path", &cfg.adaface_model_path);

    // Parse per-module log levels from JSON
    std::string log_val;
    std::vector<std::string> modules = {
        "config", "backend", "person_det", "face_det", "face_rec",
        "tracker", "face_index", "aligner", "image_util", "pipeline",
        "app", "body_attr", "face_attr"
    };
    for (const auto &mod : modules) {
        if (ExtractString(json, "log_level_" + mod, &log_val)) {
            cfg.log_levels[mod] = log_val;
        }
    }

    // 4. Initialize Logger using resolved log levels
    Logger::Instance().Initialize(cfg.log_levels);

    return cfg;
}

} // namespace face_rec
