/**
 * @file logger.cpp
 * @brief Logger and StageTimer implementation
 */

#include "logger.h"
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <algorithm>

namespace people_count {

Logger::Logger() {
    for (int i = 0; i <= static_cast<int>(LogModule::APP); ++i) {
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
#ifdef _WIN32
    localtime_s(&time_tm, &now_time_t);
#else
    localtime_r(&now_time_t, &time_tm);
#endif
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &time_tm);

    const char* filename = file;
    const char* last_slash = std::strrchr(file, '/');
    if (last_slash) {
        filename = last_slash + 1;
    }

    // Print headers
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
    }
    return "UNKNOWN";
}

const char* Logger::ModuleToString(LogModule module) {
    switch (module) {
        case LogModule::CONFIG:   return "CONFIG";
        case LogModule::BACKEND:  return "BACKEND";
        case LogModule::DETECTOR: return "DETECTOR";
        case LogModule::TRACKER:  return "TRACKER";
        case LogModule::PIPELINE: return "PIPELINE";
        case LogModule::APP:      return "APP";
    }
    return "UNKNOWN";
}

LogModule Logger::StringToModule(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "config")   return LogModule::CONFIG;
    if (lower == "backend")  return LogModule::BACKEND;
    if (lower == "detector") return LogModule::DETECTOR;
    if (lower == "tracker")  return LogModule::TRACKER;
    if (lower == "pipeline") return LogModule::PIPELINE;
    if (lower == "app")      return LogModule::APP;
    return LogModule::APP;
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

void StageTimer::Start() {
    running_ = true;
    start_time_ = std::chrono::steady_clock::now();
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

void PipelineTiming::Reset() {
    preprocess.Reset();
    body_detect.Reset();
    tracker.Reset();
    counting.Reset();
    total.Reset();
}

void PipelineTiming::Report(int interval_frames, int current_frame) const {
    if (interval_frames <= 0 || current_frame % interval_frames != 0) {
        return;
    }
    double prep = preprocess.ElapsedMs();
    double body = body_detect.ElapsedMs();
    double trk  = tracker.ElapsedMs();
    double cnt  = counting.ElapsedMs();
    double tot  = total.ElapsedMs();

    std::fprintf(stderr, "=== [Pipeline Profiling (Frame %d)] ===\n", current_frame);
    std::fprintf(stderr, "  Preprocess:  %7.2f ms\n", prep);
    std::fprintf(stderr, "  YOLO Detect: %7.2f ms\n", body);
    std::fprintf(stderr, "  ByteTracker: %7.2f ms\n", trk);
    std::fprintf(stderr, "  Counting:    %7.2f ms\n", cnt);
    std::fprintf(stderr, "  Total:       %7.2f ms\n", tot);
    std::fprintf(stderr, "=======================================\n");
    std::fflush(stderr);
}

} // namespace people_count
