// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Structured Logger
// 分级、可采样日志，不使用 std::cout 或无级别 printf。

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <string>

namespace smoking {

enum class LogLevel : int32_t {
    kWarn  = 0,
    kInfo  = 1,
    kDebug = 2,
};

// 模块标识
enum class LogModule : int32_t {
    kCommon = 0,
    kRuntime,
    kPreprocess,
    kModels,
    kPostprocess,
    kTracking,
    kEvent,
    kPipeline,
    kApp,
};

inline const char* ModuleName(LogModule m) {
    switch (m) {
        case LogModule::kCommon:    return "common";
        case LogModule::kRuntime:   return "runtime";
        case LogModule::kPreprocess:return "preprocess";
        case LogModule::kModels:    return "models";
        case LogModule::kPostprocess:return "postprocess";
        case LogModule::kTracking:  return "tracking";
        case LogModule::kEvent:     return "event";
        case LogModule::kPipeline:  return "pipeline";
        case LogModule::kApp:       return "app";
        default:                    return "unknown";
    }
}

// 全局日志级别，由 config 初始化时设置
void SetGlobalLogLevel(LogLevel level);
LogLevel GetGlobalLogLevel();

// 核心日志函数
void LogMessage(LogLevel level, LogModule module, const char* fmt, ...);

// 便捷宏
#define SMOKE_LOG_WARN(module, ...) \
    ::smoking::LogMessage(::smoking::LogLevel::kWarn, module, __VA_ARGS__)
#define SMOKE_LOG_INFO(module, ...) \
    ::smoking::LogMessage(::smoking::LogLevel::kInfo, module, __VA_ARGS__)
#define SMOKE_LOG_DEBUG(module, ...) \
    ::smoking::LogMessage(::smoking::LogLevel::kDebug, module, __VA_ARGS__)

}  // namespace smoking
