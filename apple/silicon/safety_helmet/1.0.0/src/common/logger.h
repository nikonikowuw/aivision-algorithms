/**
 * @file logger.h
 * @brief Internal logging subsystem for the safety_helmet algorithm.
 *
 * All log output goes to stderr via fprintf, prefixed with a UTC timestamp,
 * module name ([safety_helmet]), and severity level. The Engine's log
 * aggregator captures stderr from dlopen'd shared libraries.
 *
 * Macros (ALGO_LOG_INFO, ALGO_LOG_WARN, ALGO_LOG_ERROR) are the public API.
 * Never use std::cout, printf, or std::cerr directly — they would bypass
 * the structured logging format and pollute the Engine's log stream.
 *
 * ## Thread safety
 *
 * The static Log() method formats and writes atomically via fprintf.
 * While fprintf to stderr is line-buffered and not fully atomic under
 * contention, concurrent log lines from different pipeline instances are
 * acceptable — the prefixed timestamp allows correlation.
 *
 * ## Buffer size
 *
 * The internal format buffer is 1024 bytes. Messages exceeding this are
 * silently truncated by vsnprintf. This is intentional — algorithmic
 * detail belongs in the result JSON returned to the Engine, not in logs.
 */

#ifndef SAFETY_HELMET_LOGGER_H
#define SAFETY_HELMET_LOGGER_H

#include <iostream>
#include <string>
#include <chrono>
#include <cstdarg>

/**
 * @def ALGO_LOG_INFO(...)
 * @brief Log an informational message (operational milestones).
 */
#define ALGO_LOG_INFO(...) safety_helmet::Logger::Log("INFO", __VA_ARGS__)

/**
 * @def ALGO_LOG_WARN(...)
 * @brief Log a warning (recoverable issues, degraded functionality).
 */
#define ALGO_LOG_WARN(...) safety_helmet::Logger::Log("WARN", __VA_ARGS__)

/**
 * @def ALGO_LOG_ERROR(...)
 * @brief Log an error (operation failures, exceptions caught).
 */
#define ALGO_LOG_ERROR(...) safety_helmet::Logger::Log("ERROR", __VA_ARGS__)

namespace safety_helmet {

/**
 * @brief Static logging utility.
 *
 * Formats messages as:
 *   [YYYY-MM-DD HH:MM:SS] [safety_helmet] [LEVEL] message
 *
 * Uses localtime_r for thread-safe time formatting (unlike localtime which
 * returns a pointer to static storage).
 */
class Logger {
public:
    /**
     * @brief Format and emit a log message to stderr.
     *
     * @param level  Severity string ("INFO", "WARN", "ERROR").
     * @param fmt    printf-style format string.
     * @param ...    Variable arguments matching the format string.
     */
    static void Log(const char* level, const char* fmt, ...) {
        char buffer[1024];
        va_list args;
        va_start(args, fmt);
        // vsnprintf truncates gracefully if buffer is insufficient.
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        struct tm buf;
        localtime_r(&in_time_t, &buf);   // Thread-safe: stack-allocated struct tm.
        
        fprintf(stderr, "[%04d-%02d-%02d %02d:%02d:%02d] [safety_helmet] [%s] %s\n",
                buf.tm_year + 1900, buf.tm_mon + 1, buf.tm_mday,
                buf.tm_hour, buf.tm_min, buf.tm_sec,
                level, buffer);
    }
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_LOGGER_H
