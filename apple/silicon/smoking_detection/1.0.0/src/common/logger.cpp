// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Logger Implementation

#include "logger.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace smoking {

static LogLevel g_log_level = LogLevel::kWarn;

void SetGlobalLogLevel(LogLevel level) {
    g_log_level = level;
}

LogLevel GetGlobalLogLevel() {
    return g_log_level;
}

void LogMessage(LogLevel level, LogModule module, const char* fmt, ...) {
    if (level > g_log_level) return;

    const char* level_str = "[WARN] ";
    if (level == LogLevel::kInfo)  level_str = "[INFO] ";
    if (level == LogLevel::kDebug) level_str = "[DEBUG]";

    std::time_t now = std::time(nullptr);
    char time_buf[20];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));

    fprintf(stderr, "%s %s [%s] ", time_buf, level_str, ModuleName(module));

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
}

}  // namespace smoking
