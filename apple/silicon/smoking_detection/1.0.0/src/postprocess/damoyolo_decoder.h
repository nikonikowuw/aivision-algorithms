// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - DAMO-YOLO Decoder
// 解析 DAMO-YOLO 输出 tensor [1, N, 6]，字段语义通过三端一致性测试确认。

#pragma once

#include <vector>

#include "../common/types.h"

namespace smoking {

// DAMO-YOLO 输出字段语义（通过 verify_model_parity.py 确认后固定）
// [x1, y1, x2, y2, score, class_id]
struct DamoOutputParser {
    // 从模型原始输出解析检测结果
    // raw_output: [1, N, 6] 的连续浮点数据
    // num_detections: N
    // model_w, model_h: 模型输入尺寸（用于 LetterBox 坐标回映）
    // transform: 预处理坐标变换
    // 返回原图像素坐标的检测结果
    static std::vector<Detection> Parse(
        const float* raw_output,
        int32_t num_detections,
        int32_t model_w,
        int32_t model_h,
        const ImageTransform& transform);

    // 单条记录大小
    static constexpr int32_t kFieldsPerDetection = 6;
};

}  // namespace smoking
