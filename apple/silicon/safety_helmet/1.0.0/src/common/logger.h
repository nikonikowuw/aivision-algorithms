#ifndef SAFETY_HELMET_LOGGER_H
#define SAFETY_HELMET_LOGGER_H

#include <iostream>
#include <string>
#include <chrono>
#include <cstdarg>

#define ALGO_LOG_INFO(fmt, ...) safety_helmet::Logger::Log("INFO", fmt, ##__VA_ARGS__)
#define ALGO_LOG_WARN(fmt, ...) safety_helmet::Logger::Log("WARN", fmt, ##__VA_ARGS__)
#define ALGO_LOG_ERROR(fmt, ...) safety_helmet::Logger::Log("ERROR", fmt, ##__VA_ARGS__)

namespace safety_helmet {
class Logger {
public:
    static void Log(const char* level, const char* fmt, ...) {
        char buffer[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        struct tm buf;
        localtime_r(&in_time_t, &buf);
        
        fprintf(stderr, "[%04d-%02d-%02d %02d:%02d:%02d] [safety_helmet] [%s] %s\n",
                buf.tm_year + 1900, buf.tm_mon + 1, buf.tm_mday,
                buf.tm_hour, buf.tm_min, buf.tm_sec,
                level, buffer);
    }
};
}
#endif // SAFETY_HELMET_LOGGER_H
