/**
 * @file logger.h
 * @brief GPU Face Recognition — Logging system and performance timing.
 * @module Utility Layer
 */

#ifndef GPU_FACE_RECOGNITION_LOGGER_H
#define GPU_FACE_RECOGNITION_LOGGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace face_rec {

enum class LogLevel {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    NONE
};

enum class LogModule {
    CONFIG,
    BACKEND,
    PERSON_DET,
    FACE_DET,
    FACE_REC,
    TRACKER,
    FACE_INDEX,
    ALIGNER,
    IMAGE_UTIL,
    PIPELINE,
    APP,
    BODY_ATTR,
    FACE_ATTR
};

class Logger {
public:
    static Logger& Instance();

    void Initialize(const std::unordered_map<std::string, std::string>& module_levels);
    bool ShouldLog(LogModule module, LogLevel level) const;
    void Log(LogLevel level, LogModule module, const char* file, int line, const char* fmt, ...);

    static const char* LevelToString(LogLevel level);
    static const char* ModuleToString(LogModule module);
    static LogModule StringToModule(const std::string& name);
    static LogLevel StringToLevel(const std::string& name);

private:
    Logger();
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::unordered_map<LogModule, LogLevel> module_levels_;
    mutable std::mutex mutex_;
};

class StageTimer {
public:
    StageTimer() = default;
    void Start();
    void Stop();
    void Reset();
    double ElapsedMs() const;

private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
    double elapsed_ms_ = 0.0;
    bool running_ = false;
};

struct PipelineTiming {
    StageTimer person_detect;
    StageTimer tracker;
    StageTimer face_detect;
    StageTimer align;
    StageTimer extract;
    StageTimer search;
    StageTimer batch_wait;
    StageTimer total;

    void Reset();
    void Report(int interval_frames, int current_frame) const;
};

struct BlobStats {
    float min_val;
    float max_val;
    float mean_val;

    static BlobStats Compute(const float* data, size_t size);
    void LogModelIO(LogModule module, const std::string& model_name, const std::string& tensor_name, const std::vector<int64_t>& shape);
};

} // namespace face_rec

#define ALGO_LOGD(module, ...) do { \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::DEBUG)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::DEBUG, face_rec::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#define ALGO_LOGI(module, ...) do { \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::INFO)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::INFO, face_rec::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#define ALGO_LOGW(module, ...) do { \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::WARN)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::WARN, face_rec::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#define ALGO_LOGE(module, ...) do { \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::ERROR)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::ERROR, face_rec::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#endif // GPU_FACE_RECOGNITION_LOGGER_H
