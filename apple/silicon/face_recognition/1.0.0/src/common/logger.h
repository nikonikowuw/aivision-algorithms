/**
 * @file logger.h
 * @brief 人脸识别算法包日志系统与性能计时
 *        Logging system and performance timing for the face recognition package.
 * @module 基础工具层 (Utility Layer)
 * @details 提供按模块/级别过滤的结构化日志系统，支持运行时动态调整日志级别。
 *          同时包含 StageTimer（阶段计时器）和 PipelineTiming（流水线计时报告）
 *          等性能分析工具，以及 BlobStats（模型 IO 张量统计）辅助调试。
 */

#ifndef FACE_RECOGNITION_LOGGER_H
#define FACE_RECOGNITION_LOGGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace face_rec {

/**
 * @enum LogLevel
 * @brief 日志级别枚举 (Log severity levels, ascending order)
 * @details DEBUG（最详细）< INFO < WARN < ERROR < NONE（关闭日志）
 *          级别过滤规则：当日志级别 >= 模块设定的级别时才会输出。
 */
enum class LogLevel {
    DEBUG = 0,  // Detailed debugging information
    INFO,       // General informational messages
    WARN,       // Warning conditions that may need attention
    ERROR,      // Error conditions that don't halt the pipeline
    NONE        // Suppress all logging for this module
};

/**
 * @enum LogModule
 * @brief 日志模块枚举，对应算法流水线的各个组件
 *        One-to-one mapping to pipeline components for targeted log filtering.
 * @details 每个模块可独立设置日志级别，方便在生产环境中按需开启特定模块调试。
 */
enum class LogModule {
    CONFIG,      // Configuration loading module
    BACKEND,     // Inference backend (ONNX Runtime / Core ML)
    PERSON_DET,  // Person detection (SCRFD model)
    FACE_DET,    // Face detection
    FACE_REC,    // Face recognition (AdaFace feature extraction)
    TRACKER,     // Multi-object tracking (e.g. SORT / DeepSORT)
    FACE_INDEX,  // Face feature gallery / database index
    ALIGNER,     // Face alignment (landmark-based affine transform)
    IMAGE_UTIL,  // Image processing utilities
    PIPELINE,    // Main inference pipeline orchestration
    APP,         // Application-level logging
    BODY_ATTR,   // Body attribute recognition
    FACE_ATTR    // Face attribute recognition (age, gender, etc.)
};

/**
 * @class Logger
 * @brief 单例日志器，支持按模块+级别过滤的带格式日志输出
 *        Singleton logger with per-module, per-level filtering and formatted output.
 * @details 主要功能：
 *          - 每个模块独立设置日志级别（通过 config 或代码初始化）
 *          - 线程安全的日志输出（互斥锁保护）
 *          - 日志格式：[时间戳] [模块名] [级别] [文件名:行号] 消息
 *          - 通过 ALGO_LOGD/ALGO_LOGI/ALGO_LOGW/ALGO_LOGE 宏便捷调用
 */
class Logger {
public:
    /**
     * @brief 获取全局单例实例
     * @return Logger 引用
     */
    static Logger& Instance();

    /**
     * @brief 初始化日志器，设置各模块的日志级别
     * @param module_levels 模块名称到级别名称的映射表
     *                      例如：{"face_det": "DEBUG", "tracker": "INFO"}
     */
    void Initialize(const std::unordered_map<std::string, std::string>& module_levels);

    /**
     * @brief 判断指定模块/级别组合是否应输出日志
     * @param module 目标模块
     * @param level  目标级别
     * @return true 如果 level >= 该模块设定的级别阈值
     */
    bool ShouldLog(LogModule module, LogLevel level) const;

    /**
     * @brief 格式化日志输出（printf 风格）
     * @param level   日志级别
     * @param module  日志模块
     * @param file    源文件名（通常传入 __FILE__）
     * @param line    行号（通常传入 __LINE__）
     * @param fmt     printf 风格格式字符串
     * @param ...     变长参数列表
     */
    void Log(LogLevel level, LogModule module, const char* file, int line, const char* fmt, ...);

    // 枚举值与字符串双向转换工具
    static const char* LevelToString(LogLevel level);        // LogLevel → string for display
    static const char* ModuleToString(LogModule module);      // LogModule → string for display
    static LogModule StringToModule(const std::string& name);  // String → LogModule (case-insensitive)
    static LogLevel StringToLevel(const std::string& name);    // String → LogLevel (case-insensitive)

private:
    Logger();                           // Singleton: private constructor
    ~Logger() = default;                // Implicitly kept private
    Logger(const Logger&) = delete;     // Non-copyable
    Logger& operator=(const Logger&) = delete;  // Non-assignable

    std::unordered_map<LogModule, LogLevel> module_levels_;  // Per-module log level map
    mutable std::mutex mutex_;          // Mutex for thread-safe log output
};

/**
 * @class StageTimer
 * @brief 阶段计时器，用于测量算法流水线中各个阶段的耗时
 *        Accumulating timer for profiling individual pipeline stages.
 * @details 支持累积计时：多次 Start/Stop 循环会将时间累加。
 *          正在运行中的计时器也可查询当前耗时（不停止计时）。
 */
class StageTimer {
public:
    StageTimer() = default;

    /**
     * @brief 开始计时
     */
    void Start();

    /**
     * @brief 停止计时，将本次耗时累加到 elapsed_ms_ 中
     */
    void Stop();

    /**
     * @brief 重置累积耗时（清零并设为非运行状态）
     */
    void Reset();

    /**
     * @brief 获取当前累积耗时（毫秒）
     * @return 如果正在运行中，返回已累积时间 + 当前段已耗时
     */
    double ElapsedMs() const;

private:
    std::chrono::steady_clock::time_point start_time_;  // Last start timestamp
    std::chrono::steady_clock::time_point end_time_;    // Last stop timestamp
    double elapsed_ms_ = 0.0;                            // Accumulated elapsed time (ms)
    bool running_ = false;                               // Whether timer is actively ticking
};

/**
 * @struct PipelineTiming
 * @brief 算法流水线各阶段计时器集合
 *        Collection of StageTimers for the end-to-end inference pipeline.
 * @details 包含从人体检测到属性识别的全流程各阶段计时器，
 *          支持按帧间隔输出性能报告，用于线上监控和调优。
 */
struct PipelineTiming {
    StageTimer body_detect;    // 人体检测阶段耗时
    StageTimer tracker;        // 多目标跟踪阶段耗时
    StageTimer face_detect;    // 人脸检测阶段耗时
    StageTimer align;          // 人脸对齐阶段耗时
    StageTimer extract;        // 特征提取阶段耗时
    StageTimer search;         // 数据库检索阶段耗时
    StageTimer attr_extract;   // 属性识别阶段耗时
    StageTimer total;          // 整条流水线总耗时

    /**
     * @brief 重置所有计时器
     */
    void Reset();

    /**
     * @brief 输出性能报告（按帧间隔）
     * @param interval_frames 每隔多少帧报告一次
     * @param current_frame   当前帧编号
     */
    void Report(int interval_frames, int current_frame) const;
};

/**
 * @struct BlobStats
 * @brief 模型输入/输出张量统计信息
 *        Statistics for model input/output tensor blobs.
 * @details 计算浮点张量的最小值、最大值、均值，
 *          用于调试模型输出是否正常（如 NaN 检测、数值漂移监控）。
 */
struct BlobStats {
    float min_val;   // Minimum value in the tensor
    float max_val;   // Maximum value in the tensor
    float mean_val;  // Mean (average) value of all elements

    /**
     * @brief 计算 float 数组的统计量
     * @param data 数据指针
     * @param size 元素个数
     * @return 包含 min/max/mean 的统计结果
     */
    static BlobStats Compute(const float* data, size_t size);

    /**
     * @brief 以 DEBUG 级别记录模型 IO 统计信息
     * @param module     日志模块
     * @param model_name 模型名称（如 "scrfd_500m"）
     * @param tensor_name 张量名称
     * @param shape      张量形状
     */
    void LogModelIO(LogModule module, const std::string& model_name, const std::string& tensor_name, const std::vector<int64_t>& shape);
};

} // namespace face_rec

/**
 * @def ALGO_LOGD
 * @brief DEBUG 级别日志宏，自动传递文件名和行号
 * @param module 日志模块枚举成员（不包含 LogModule:: 前缀）
 * @param fmt    printf 风格格式字符串
 * @param ...    变长参数
 * @note 使用方式：ALGO_LOGD(FACE_DET, "confidence: %.2f", conf);
 */
#define ALGO_LOGD(module, fmt, ...) \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::DEBUG)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::DEBUG, face_rec::LogModule::module, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

/**
 * @def ALGO_LOGI
 * @brief INFO 级别日志宏
 */
#define ALGO_LOGI(module, fmt, ...) \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::INFO)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::INFO, face_rec::LogModule::module, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

/**
 * @def ALGO_LOGW
 * @brief WARN 级别日志宏
 */
#define ALGO_LOGW(module, fmt, ...) \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::WARN)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::WARN, face_rec::LogModule::module, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

/**
 * @def ALGO_LOGE
 * @brief ERROR 级别日志宏
 */
#define ALGO_LOGE(module, fmt, ...) \
    if (face_rec::Logger::Instance().ShouldLog(face_rec::LogModule::module, face_rec::LogLevel::ERROR)) \
        face_rec::Logger::Instance().Log(face_rec::LogLevel::ERROR, face_rec::LogModule::module, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#endif // FACE_RECOGNITION_LOGGER_H
