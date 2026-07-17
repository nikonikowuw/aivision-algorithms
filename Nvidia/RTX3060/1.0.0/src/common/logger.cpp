/**
 * @file logger.cpp
 * @brief GPU Face Recognition — Logging system implementation.
 */

#include "logger.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <algorithm>

namespace face_rec {

Logger::Logger() {
    for (int i = 0; i <= static_cast<int>(LogModule::FACE_ATTR); ++i) {
        module_levels_[static_cast<LogModule>(i)] = LogLevel::INFO;
    }
}

Logger& Logger::Instance() {
    static Logger instance;
    return instance;
}

void Logger::Initialize(const std::unordered_map<std::string, std::string>& module_levels) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : module_levels) {
        LogModule module = StringToModule(pair.first);
        LogLevel level = StringToLevel(pair.second);
        module_levels_[module] = level;
    }
}

bool Logger::ShouldLog(LogModule module, LogLevel level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = module_levels_.find(module);
    if (it != module_levels_.end()) {
        return level >= it->second;
    }
    return level >= LogLevel::INFO;
}

void Logger::Log(LogLevel level, LogModule module, const char* file, int line, const char* fmt, ...) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    char time_buf[64];
    struct tm time_tm;
    localtime_r(&now_time_t, &time_tm);
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &time_tm);

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

    std::fprintf(stderr, "[%s.%03d] [%s] [%s] [%s:%d] ",
                 time_buf, static_cast<int>(now_ms.count()),
                 ModuleToString(module), LevelToString(level),
                 filename, line);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

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

// StageTimer

void StageTimer::Start() {
    start_time_ = std::chrono::steady_clock::now();
    running_ = true;
}

void StageTimer::Stop() {
    if (running_) {
        end_time_ = std::chrono::steady_clock::now();
        elapsed_ms_ += std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
        running_ = false;
    }
}

void StageTimer::Reset() {
    elapsed_ms_ = 0.0;
    running_ = false;
}

double StageTimer::ElapsedMs() const {
    if (running_) {
        auto now = std::chrono::steady_clock::now();
        return elapsed_ms_ + std::chrono::duration<double, std::milli>(now - start_time_).count();
    }
    return elapsed_ms_;
}

// PipelineTiming

void PipelineTiming::Reset() {
    person_detect.Reset();
    tracker.Reset();
    face_detect.Reset();
    align.Reset();
    extract.Reset();
    search.Reset();
    batch_wait.Reset();
    total.Reset();
}

void PipelineTiming::Report(int interval_frames, int current_frame) const {
    if (interval_frames <= 0 || current_frame % interval_frames != 0) return;
    std::fprintf(stderr, "=== Pipeline Timing Report at frame %d ===\n", current_frame);
    std::fprintf(stderr, "  Person Detection : %.2f ms\n", person_detect.ElapsedMs());
    std::fprintf(stderr, "  Tracker          : %.2f ms\n", tracker.ElapsedMs());
    std::fprintf(stderr, "  Face Detection   : %.2f ms\n", face_detect.ElapsedMs());
    std::fprintf(stderr, "  Face Alignment   : %.2f ms\n", align.ElapsedMs());
    std::fprintf(stderr, "  Feature Extract  : %.2f ms\n", extract.ElapsedMs());
    std::fprintf(stderr, "  Gallery Search   : %.2f ms\n", search.ElapsedMs());
    std::fprintf(stderr, "  Batch Wait       : %.2f ms\n", batch_wait.ElapsedMs());
    std::fprintf(stderr, "  Total Pipeline   : %.2f ms\n", total.ElapsedMs());
    std::fprintf(stderr, "===========================================\n");
    std::fflush(stderr);
}

// BlobStats

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

void BlobStats::LogModelIO(LogModule module, const std::string& model_name, const std::string& tensor_name, const std::vector<int64_t>& shape) {
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size()) shape_str += ",";
    }
    shape_str += "]";

    if (Logger::Instance().ShouldLog(module, LogLevel::DEBUG)) {
        Logger::Instance().Log(LogLevel::DEBUG, module, __FILE__, __LINE__,
                               "Model IO: Model=%s, Tensor=%s, Shape=%s, Min=%.6f, Max=%.6f, Mean=%.6f",
                               model_name.c_str(), tensor_name.c_str(), shape_str.c_str(), min_val, max_val, mean_val);
    }
}

} // namespace face_rec
