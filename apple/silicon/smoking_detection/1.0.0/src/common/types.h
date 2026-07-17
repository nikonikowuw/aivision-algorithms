// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Common Types
// 所有 pipeline 中间框统一使用原图像素坐标，仅最终输出归一化。

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace smoking {

// ---------------------------------------------------------------------------
// Error codes — 对应 ABI 入口返回值，不使用异常表达正常失败
// ---------------------------------------------------------------------------
enum class ErrorCode : int32_t {
    kSuccess              =   0,
    kInvalidParam         =  -1,
    kInvalidBuffer        =  -2,
    kPreprocessFailed     =  -3,
    kInferenceFailed      =  -4,
    kPostprocessFailed    =  -5,
    kSerializationFailed  =  -6,
    kUnsupportedCapability = -7,
    kInternalError        = -127,
};

// ---------------------------------------------------------------------------
// Geometric primitives — 原图像素坐标系
// ---------------------------------------------------------------------------
struct PointF {
    float x = 0.0f;
    float y = 0.0f;
};

// Rect — 原图像素坐标，(x, y) 为左上角
struct RectF {
    float x      = 0.0f;
    float y      = 0.0f;
    float width  = 0.0f;
    float height = 0.0f;

    float Left()   const { return x; }
    float Top()    const { return y; }
    float Right()  const { return x + width; }
    float Bottom() const { return y + height; }
    PointF Center() const { return { x + width * 0.5f, y + height * 0.5f }; }
    float Area()    const { return width * height; }

    // 裁剪到 [0, frame_w] x [0, frame_h]
    void ClipToFrame(float frame_w, float frame_h) {
        if (x < 0.0f) { width += x; x = 0.0f; }
        if (y < 0.0f) { height += y; y = 0.0f; }
        if (x + width > frame_w)  width  = frame_w - x;
        if (y + height > frame_h) height = frame_h - y;
        if (width < 0.0f) width = 0.0f;
        if (height < 0.0f) height = 0.0f;
    }
};

// NormalizedRect — 归一化 [0,1]，用于最终 JSON 输出
struct NormalizedRect {
    float x      = 0.0f;
    float y      = 0.0f;
    float width  = 0.0f;
    float height = 0.0f;
};

// ---------------------------------------------------------------------------
// ImageTransform — 预处理返回的不可变坐标映射对象
// 后处理只能通过该对象把模型框转换为原图像素框
// ---------------------------------------------------------------------------
struct ImageTransform {
    RectF source_region_in_frame;  // 原图中被裁剪的区域
    float scale      = 1.0f;       // resize 缩放比
    float pad_x      = 0.0f;       // letterbox X 填充
    float pad_y      = 0.0f;       // letterbox Y 填充
    int32_t model_width  = 640;    // 模型输入宽
    int32_t model_height = 640;    // 模型输入高
    int32_t frame_width  = 0;      // 原图宽
    int32_t frame_height = 0;      // 原图高

    // 模型坐标 → 原图像素坐标
    RectF ModelToOriginal(const RectF& model_rect) const {
        float ox = (model_rect.x - pad_x) / scale + source_region_in_frame.x;
        float oy = (model_rect.y - pad_y) / scale + source_region_in_frame.y;
        float ow = model_rect.width / scale;
        float oh = model_rect.height / scale;
        RectF r{ox, oy, ow, oh};
        r.ClipToFrame(static_cast<float>(frame_width),
                      static_cast<float>(frame_height));
        return r;
    }
};

// ---------------------------------------------------------------------------
// Detection — 单次检测结果（原图像素坐标）
// ---------------------------------------------------------------------------
struct Detection {
    RectF   bbox;
    float   confidence = 0.0f;
    int32_t class_id   = 0;
};

// ---------------------------------------------------------------------------
// TrackedPerson — 跟踪后的人员
// ---------------------------------------------------------------------------
struct TrackedPerson {
    int32_t track_id    = -1;
    RectF   bbox;               // 人体框，原图像素坐标
    float   confidence  = 0.0f;
    int64_t last_update_ms = 0; // wall-clock 最后更新时间
    int64_t first_seen_ms  = 0;
};

// ---------------------------------------------------------------------------
// SmokingEvidence — 单次香烟检出证据
// ---------------------------------------------------------------------------
struct SmokingEvidence {
    RectF   cigarette_bbox;     // 香烟框，原图像素坐标
    float   confidence   = 0.0f;
    int32_t track_id     = -1;  // 关联的人物 track_id
};

// ---------------------------------------------------------------------------
// SmokingEvent — 最终输出事件
// ---------------------------------------------------------------------------
struct SmokingEvent {
    int32_t  category_code   = 14001;
    float    detect_confidence = 0.0f;
    RectF    person_bbox;       // 吸烟人物框
    RectF    cigarette_bbox;   // 触发事件的香烟证据框
    int32_t  track_id          = -1;
    std::string event_id;
};

// ---------------------------------------------------------------------------
// Pixel format helpers
// ---------------------------------------------------------------------------
constexpr uint32_t kPixelFormatBGR24 = 0;
constexpr uint32_t kPixelFormatBGRA = 0x42475241;  // 'BGRA'
constexpr uint32_t kPixelFormat420YpCbCr8BiPlanarVideoRange = 0x34323076;  // '420v'
constexpr uint32_t kPixelFormat420YpCbCr8BiPlanarFullRange = 0x34323066;   // '420f'
constexpr uint32_t kPixelFormatNV12VR = kPixelFormat420YpCbCr8BiPlanarVideoRange;
constexpr uint32_t kPixelFormatNV12FR = kPixelFormat420YpCbCr8BiPlanarFullRange;

// ---------------------------------------------------------------------------
// Validity helpers
// ---------------------------------------------------------------------------
inline bool IsFinite(float v) {
    return std::isfinite(v);
}

inline bool IsValidRect(const RectF& r) {
    return IsFinite(r.x) && IsFinite(r.y) &&
           IsFinite(r.width) && IsFinite(r.height) &&
           r.width >= 0.0f && r.height >= 0.0f;
}

inline bool NormalizeRect(const RectF& source, int32_t frame_width,
                          int32_t frame_height, NormalizedRect& normalized) {
    if (frame_width <= 0 || frame_height <= 0 || !IsValidRect(source)) {
        return false;
    }

    RectF clipped = source;
    clipped.ClipToFrame(static_cast<float>(frame_width),
                        static_cast<float>(frame_height));
    if (!IsValidRect(clipped)) return false;

    normalized.x = std::clamp(clipped.x / frame_width, 0.0f, 1.0f);
    normalized.y = std::clamp(clipped.y / frame_height, 0.0f, 1.0f);
    normalized.width = std::clamp(clipped.width / frame_width, 0.0f, 1.0f);
    normalized.height = std::clamp(clipped.height / frame_height, 0.0f, 1.0f);
    return IsFinite(normalized.x) && IsFinite(normalized.y) &&
           IsFinite(normalized.width) && IsFinite(normalized.height);
}

}  // namespace smoking
