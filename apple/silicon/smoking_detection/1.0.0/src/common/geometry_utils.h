// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Geometry Utilities
// IoU 计算、检测区域过滤等几何工具。

#pragma once

#include <cmath>

#include "types.h"

namespace smoking {

// 计算两个框的 IoU
inline float ComputeIoU(const RectF& a, const RectF& b) {
    float x1 = std::max(a.Left(),   b.Left());
    float y1 = std::max(a.Top(),    b.Top());
    float x2 = std::min(a.Right(),  b.Right());
    float y2 = std::min(a.Bottom(), b.Bottom());

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = a.Area() + b.Area() - inter;
    if (union_area <= 0.0f) return 0.0f;
    return inter / union_area;
}

// 计算两个框的中心距离（归一化到较大框的对角线）
inline float CenterDistanceNormalized(const RectF& a, const RectF& b) {
    float cx_a = a.x + a.width * 0.5f;
    float cy_a = a.y + a.height * 0.5f;
    float cx_b = b.x + b.width * 0.5f;
    float cy_b = b.y + b.height * 0.5f;
    float dx = cx_a - cx_b;
    float dy = cy_a - cy_b;
    float dist = std::sqrt(dx * dx + dy * dy);
    float diag = std::sqrt(a.width * a.width + a.height * a.height);
    if (diag <= 0.0f) return 1.0f;
    return dist / diag;
}

// 判断点是否在多边形内（射线法）
inline bool PointInPolygon(float px, float py,
                           const std::vector<PointF>& polygon) {
    if (polygon.size() < 3) return false;
    bool inside = false;
    size_t n = polygon.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        float xi = polygon[i].x, yi = polygon[i].y;
        float xj = polygon[j].x, yj = polygon[j].y;
        if (((yi > py) != (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

// 判断框是否与检测区域相交（中心点在区域内，或框与区域有交集）
inline bool RectIntersectsRegion(const RectF& rect,
                                 const std::vector<PointF>& region) {
    PointF c = rect.Center();
    if (PointInPolygon(c.x, c.y, region)) return true;
    // 检查区域顶点是否在框内
    for (const auto& p : region) {
        if (p.x >= rect.Left() && p.x <= rect.Right() &&
            p.y >= rect.Top()  && p.y <= rect.Bottom()) {
            return true;
        }
    }
    return false;
}

}  // namespace smoking
