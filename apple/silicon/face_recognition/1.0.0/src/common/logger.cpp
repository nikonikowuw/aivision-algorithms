/**
 * @file logger.cpp
 * @brief 日志系统与性能计时实现
 *        Implementation of the logging system and performance timing utilities.
 * @module 基础工具层 (Utility Layer)
 * @details 实现了按模块/级别过滤的结构化日志系统，
 *          以及 StageTimer（阶段计时器）、PipelineTiming（流水线计时报告）、
 *          BlobStats（模型 IO 张量统计）等性能分析工具的所有功能。
 */

#include "logger.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <algorithm>

namespace face_rec {

/**
 * @brief 构造函数：将所有模块的日志级别初始化为 INFO
 *        Default constructor — all modules start at INFO level.
 */
Logger::Logger() {
    // Default levels for all modules
    for (int i = 0; i <= static_cast<int>(LogModule::FACE_ATTR); ++i) {
        module_levels_[static_cast<LogModule>(i)] = LogLevel::INFO;
    }
}

/**
 * @brief 获取全局单例实例（Meyer's Singleton）
 * @return Logger 引用
 */
Logger& Logger::Instance() {
    static Logger instance;
    return instance;
}

/**
 * @brief 初始化日志器，根据配置映射表设置各模块的日志级别
 * @param module_levels 模块名称 → 级别名称的映射
 */
void Logger::Initialize(const std::unordered_map<std::string, std::string>& module_levels) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : module_levels) {
        LogModule module = StringToModule(pair.first);
        LogLevel level = StringToLevel(pair.second);
        module_levels_[module] = level;
    }
}

/**
 * @brief 判断指定模块/级别组合是否应该输出日志
 * @param module 模块枚举值
 * @param level  级别枚举值
 * @return true 当 level >= 模块设定的级别阈值；未设定时默认 INFO
 */
bool Logger::ShouldLog(LogModule module, LogLevel level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = module_levels_.find(module);
    if (it != module_levels_.end()) {
        return level >= it->second;
    }
    return level >= LogLevel::INFO;
}

/**
 * @brief 带时间戳和文件位置的格式化日志输出
 *        Log a formatted message with timestamp, module tag, and source location.
 * @details 输出格式：[YYYY-MM-DD HH:MM:SS.mmm] [MODULE] [LEVEL] [filename:line] message
 *          所有输出写入 stderr 并立即刷新，确保日志不会因缓冲区而丢失。
 * @param level  日志级别
 * @param module 日志模块
 * @param file   源文件路径（自动截取文件名部分）
 * @param line   行号
 * @param fmt    格式字符串
 * @param ...    变长参数
 */
void Logger::Log(LogLevel level, LogModule module, const char* file, int line, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get current time with millisecond precision
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    char time_buf[64];
    struct tm time_tm;
#ifdef _WIN32
    localtime_s(&time_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &time_tm);
#endif
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &time_tm);

    // Extract just the filename from full path
    const char* filename = file;
    const char* last_slash = std::strrchr(file, '/');
    if (last_slash) {
        filename = last_slash + 1;
    } else {
        last_slash = std::strrchr(file, '\\');
        if (last_slash) {
            filename = last_slash + 1;
        }
    }

    // Write log prefix: timestamp, module, level, source location
    std::fprintf(stderr, "[%s.%03d] [%s] [%s] [%s:%d] ", 
                 time_buf, static_cast<int>(now_ms.count()),
                 ModuleToString(module), LevelToString(level),
                 filename, line);

    // Write user message
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

/**
 * @brief 将日志级别枚举值转换为可读字符串
 * @param level 日志级别枚举
 * @return 对应的字符串常量（如 "DEBUG", "INFO" 等）
 */
const char* Logger::LevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::NONE:  return "NONE";
        default:              return "UNKNOWN";
    }
}

/**
 * @brief 将日志模块枚举值转换为可读字符串
 * @param module 日志模块枚举
 * @return 对应的字符串常量（如 "FACE_DET", "TRACKER" 等）
 */
const char* Logger::ModuleToString(LogModule module) {
    switch (module) {
        case LogModule::CONFIG:     return "CONFIG";
        case LogModule::BACKEND:    return "BACKEND";
        case LogModule::PERSON_DET: return "PERSON_DET";
        case LogModule::FACE_DET:   return "FACE_DET";
        case LogModule::FACE_REC:   return "FACE_REC";
        case LogModule::TRACKER:    return "TRACKER";
        case LogModule::FACE_INDEX: return "FACE_INDEX";
        case LogModule::ALIGNER:    return "ALIGNER";
        case LogModule::IMAGE_UTIL: return "IMAGE_UTIL";
        case LogModule::PIPELINE:   return "PIPELINE";
        case LogModule::APP:        return "APP";
        case LogModule::BODY_ATTR:  return "BODY_ATTR";
        case LogModule::FACE_ATTR:  return "FACE_ATTR";
        default:                    return "UNKNOWN";
    }
}

/**
 * @brief 将字符串转换为日志模块枚举（大小写不敏感）
 *        Convert string to LogModule enum (case-insensitive).
 * @param name 模块名称字符串
 * @return 对应的 LogModule 枚举值，无法识别时返回 PIPELINE
 */
LogModule Logger::StringToModule(const std::string& name) {
    std::string upper = name;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    if (upper == "CONFIG")     return LogModule::CONFIG;
    if (upper == "BACKEND")    return LogModule::BACKEND;
    if (upper == "PERSON_DET") return LogModule::PERSON_DET;
    if (upper == "FACE_DET")   return LogModule::FACE_DET;
    if (upper == "FACE_REC")   return LogModule::FACE_REC;
    if (upper == "TRACKER")    return LogModule::TRACKER;
    if (upper == "FACE_INDEX") return LogModule::FACE_INDEX;
    if (upper == "ALIGNER")    return LogModule::ALIGNER;
    if (upper == "IMAGE_UTIL") return LogModule::IMAGE_UTIL;
    if (upper == "PIPELINE")   return LogModule::PIPELINE;
    if (upper == "APP")        return LogModule::APP;
    if (upper == "BODY_ATTR")  return LogModule::BODY_ATTR;
    if (upper == "FACE_ATTR")  return LogModule::FACE_ATTR;
    return LogModule::PIPELINE;
}

/**
 * @brief 将字符串转换为日志级别枚举（大小写不敏感）
 *        Convert string to LogLevel enum (case-insensitive).
 * @param name 级别名称字符串
 * @return 对应的 LogLevel 枚举值，无法识别时返回 INFO
 */
LogLevel Logger::StringToLevel(const std::string& name) {
    std::string upper = name;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    if (upper == "DEBUG") return LogLevel::DEBUG;
    if (upper == "INFO")  return LogLevel::INFO;
    if (upper == "WARN")  return LogLevel::WARN;
    if (upper == "ERROR") return LogLevel::ERROR;
    if (upper == "NONE")  return LogLevel::NONE;
    return LogLevel::INFO;
}

// StageTimer Implementation

/**
 * @brief 开始计时（记录当前时间点）
 */
void StageTimer::Start() {
    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
}

/**
 * @brief 停止计时，将本次耗时累加到总累积时间中
 *        Stop the timer and accumulate the elapsed time.
 */
void StageTimer::Stop() {
    if (running_) {
        end_time_ = std::chrono::steady_clock::now();
        elapsed_ms_ += std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
        running_ = false;
    }
}

/**
 * @brief 重置计时器（清零累积时间并设为非运行状态）
 */
void StageTimer::Reset() {
    elapsed_ms_ = 0.0;
    running_ = false;
}

/**
 * @brief 获取当前累积耗时
 * @return 累积毫秒数。如果正在计时中，返回已累积值 + 当前段已流逝时间
 */
double StageTimer::ElapsedMs() const {
    if (running_) {
        auto now = std::chrono::steady_clock::now();
        return elapsed_ms_ + std::chrono::duration<double, std::milli>(now - start_time_).count();
    }
    return elapsed_ms_;
}

// PipelineTiming Implementation

/**
 * @brief 重置流水线中所有阶段计时器
 */
void PipelineTiming::Reset() {
    body_detect.Reset();
    tracker.Reset();
    face_detect.Reset();
    align.Reset();
    extract.Reset();
    search.Reset();
    attr_extract.Reset();
    total.Reset();
}

/**
 * @brief 按指定帧间隔输出流水线性能报告
 *        Print a pipeline timing report at a fixed frame interval.
 * @details 报告内容包括人体检测、跟踪、人脸检测、对齐、特征提取、
 *          数据库检索、属性识别以及总耗时，单位均为毫秒。
 * @param interval_frames 输出间隔（帧数）
 * @param current_frame   当前帧编号
 */
void PipelineTiming::Report(int interval_frames, int current_frame) const {
    if (interval_frames <= 0 || current_frame % interval_frames != 0) return;
    std::fprintf(stderr, "=== Pipeline Timing Report at frame %d ===\n", current_frame);
    std::fprintf(stderr, "  Body Detection  : %.2f ms\n", body_detect.ElapsedMs());
    std::fprintf(stderr, "  Tracker         : %.2f ms\n", tracker.ElapsedMs());
    std::fprintf(stderr, "  Face Detection  : %.2f ms\n", face_detect.ElapsedMs());
    std::fprintf(stderr, "  Face Alignment  : %.2f ms\n", align.ElapsedMs());
    std::fprintf(stderr, "  Face Feature Ext: %.2f ms\n", extract.ElapsedMs());
    std::fprintf(stderr, "  Database Search : %.2f ms\n", search.ElapsedMs());
    std::fprintf(stderr, "  Attr Extraction : %.2f ms\n", attr_extract.ElapsedMs());
    std::fprintf(stderr, "  Total Pipeline  : %.2f ms\n", total.ElapsedMs());
    std::fprintf(stderr, "===========================================\n");
    std::fflush(stderr);
}

// BlobStats Implementation

/**
 * @brief 计算浮点张量的统计量（最小值、最大值、均值）
 *        Compute min, max, and mean statistics of a float array.
 * @param data 浮点数据指针
 * @param size 数据元素个数
 * @return 包含 min/max/mean 的统计结果；size=0 时返回全零
 */
BlobStats BlobStats::Compute(const float* data, size_t size) {
    if (size == 0) return {0.0f, 0.0f, 0.0f};
    float min_val = data[0];
    float max_val = data[0];
    double sum_val = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }
    return {min_val, max_val, static_cast<float>(sum_val / size)};
}

/**
 * @brief 以 DEBUG 级别记录模型输入/输出张量的统计信息
 *        Log model I/O tensor statistics at DEBUG level.
 * @details 包含模型名称、张量名称、形状（shape）以及
 *          最小值/最大值/均值的统计值，便于调试模型输出数值范围是否异常。
 * @param module     所属日志模块
 * @param model_name 模型名称
 * @param tensor_name 张量名称
 * @param shape      张量形状（维度信息）
 */
void BlobStats::LogModelIO(LogModule module, const std::string& model_name, const std::string& tensor_name, const std::vector<int64_t>& shape) {
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size()) shape_str += ",";
    }
    shape_str += "]";
    
    if (Logger::Instance().ShouldLog(module, LogLevel::DEBUG)) {
        Logger::Instance().Log(LogLevel::DEBUG, module, __FILE__, __LINE__,
                               "Model IO Statistics: Model=%s, Tensor=%s, Shape=%s, Min=%.6f, Max=%.6f, Mean=%.6f",
                               model_name.c_str(), tensor_name.c_str(), shape_str.c_str(), min_val, max_val, mean_val);
    }
}

} // namespace face_rec
