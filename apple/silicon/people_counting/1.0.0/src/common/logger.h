/**
 * @file logger.h
 * @brief Logging system and performance timing
 */

#ifndef PEOPLE_COUNTING_LOGGER_H
#define PEOPLE_COUNTING_LOGGER_H

#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace people_count {

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
    DETECTOR,
    TRACKER,
    PIPELINE,
    APP
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
    StageTimer preprocess;
    StageTimer body_detect;
    StageTimer tracker;
    StageTimer counting;
    StageTimer total;

    void Reset();
    void Report(int interval_frames, int current_frame) const;
};

} // namespace people_count

#define ALGO_LOGD(module, ...) do { \
    if (people_count::Logger::Instance().ShouldLog(people_count::LogModule::module, people_count::LogLevel::DEBUG)) \
        people_count::Logger::Instance().Log(people_count::LogLevel::DEBUG, people_count::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#define ALGO_LOGI(module, ...) do { \
    if (people_count::Logger::Instance().ShouldLog(people_count::LogModule::module, people_count::LogLevel::INFO)) \
        people_count::Logger::Instance().Log(people_count::LogLevel::INFO, people_count::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#define ALGO_LOGW(module, ...) do { \
    if (people_count::Logger::Instance().ShouldLog(people_count::LogModule::module, people_count::LogLevel::WARN)) \
        people_count::Logger::Instance().Log(people_count::LogLevel::WARN, people_count::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#define ALGO_LOGE(module, ...) do { \
    if (people_count::Logger::Instance().ShouldLog(people_count::LogModule::module, people_count::LogLevel::ERROR)) \
        people_count::Logger::Instance().Log(people_count::LogLevel::ERROR, people_count::LogModule::module, __FILE__, __LINE__, __VA_ARGS__); \
} while (0)

#endif // PEOPLE_COUNTING_LOGGER_H
