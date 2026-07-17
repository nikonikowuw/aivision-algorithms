// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Configuration
// 配置由 detector_init(config_json) 解析一次，使用结构化 JSON parser。

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "types.h"

namespace smoking {

// ---------------------------------------------------------------------------
// 检测区域 — 归一化多边形，支持一个或多个
// ---------------------------------------------------------------------------
struct DetectionRegion {
    std::vector<PointF> points;  // 归一化顶点，面积必须 > 0
};

// ---------------------------------------------------------------------------
// AlgoConfig — 所有可配置参数，与 algo_meta.yaml / C++ 默认值保持一致
// ---------------------------------------------------------------------------
struct AlgoConfig {
    // --- 推理后端 ---
    std::string backend = "coreml_native";  // coreml_native | ort_coreml

    // --- 分析帧率 ---
    float analysis_fps = 4.0f;  // 每秒分析帧数

    // --- 人体检测 ---
    int32_t max_persons          = 10;
    float   person_conf_threshold = 0.50f;
    float   person_nms_threshold  = 0.70f;

    // --- 香烟检测 ---
    float cigarette_conf_threshold = 0.60f;
    float cigarette_nms_threshold  = 0.70f;

    // --- 分块检测 ---
    bool    tile_enabled         = true;
    int32_t tile_grid_rows       = 2;
    int32_t tile_grid_cols       = 2;
    float   tile_overlap         = 0.15f;
    int32_t tile_budget_per_tick = 2;

    // --- 人物上部 ROI ---
    float upper_body_ratio = 0.60f;  // 人物框顶部至该比例
    float roi_expand_x     = 0.15f;  // 左右扩展
    float roi_expand_top   = 0.05f;  // 顶部扩展

    // --- 时序状态机 ---
    int32_t temporal_window  = 5;      // 最近 N 次评估
    int32_t confirm_hits     = 3;      // 至少 M 次 hit 确认
    int64_t rearm_ms         = 2000;   // 连续无证据后重新布防

    // --- 跟踪器 ---
    float   tracker_iou_threshold = 0.20f;
    int64_t tracker_max_lost_ms   = 2000;

    // --- 检测区域 ---
    std::vector<DetectionRegion> detection_regions;

    // --- 模型路径 ---
    std::string model_dir = "weights";  // 相对动态库目录解析
    std::string runtime_root;            // 由 C ABI 层设置，不从 JSON 读取

    // --- 日志 ---
    int32_t log_level = 0;  // 0=warn, 1=info, 2=debug
};

// ---------------------------------------------------------------------------
// Config parsing — 解析 JSON 配置字符串
// 返回 ErrorCode::kSuccess 表示成功
// ---------------------------------------------------------------------------
ErrorCode ParseConfig(const char* config_json, AlgoConfig& config,
                      std::string& error_msg);

// ---------------------------------------------------------------------------
// Config validation — 校验参数范围和一致性
// ---------------------------------------------------------------------------
ErrorCode ValidateConfig(const AlgoConfig& config, std::string& error_msg);

}  // namespace smoking
