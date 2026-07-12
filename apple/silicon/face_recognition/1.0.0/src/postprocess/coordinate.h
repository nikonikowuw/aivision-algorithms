/**
 * @file coordinate.h
 * @brief 坐标映射与 ROI 计算类声明
 *        Declarations for coordinate mapping and ROI computation
 *
 * 提供 Letterbox→原图的两阶段坐标映射、自适应头部 ROI 裁剪和 BBox 归一化功能。
 * Provides two-stage letterbox→original coordinate mapping, adaptive head ROI
 * cropping, and bbox normalization.
 */
#ifndef FACE_RECOGNITION_COORDINATE_H
#define FACE_RECOGNITION_COORDINATE_H

#include "common/types.h"
#include "common/config.h"
#include "preprocess/image_utils.h"
#include <vector>

namespace face_rec {

/**
 * @brief 坐标映射器 / Coordinate mapping utility
 *
 * 负责将检测结果从 Letterbox 归一化坐标空间映射回原始图像坐标。
 * 支持单阶段（直接 Letterbox→原图）和两阶段（Letterbox→ROI→原图）两种模式。
 * Maps detection results from letterbox normalized coordinates back to the
 * original image. Supports both 1-stage (direct letterbox→original) and
 * 2-stage (letterbox→ROI→original) pipelines.
 */
class CoordinateMapper {
public:
    /**
     * @brief 单阶段映射：Letterbox(640) → 原始帧坐标
     *        1-stage: Map from letterbox(640) to original frame coordinates
     *
     * 适用于直接在全图尺度（640 输入）上检测后，将 bbox 和关键点映射回原图。
     * Used when detection runs at full-image scale (640 input), mapping bbox
     * and landmarks back to the original image.
     */
    static void RestoreToOriginal(
        const image_utils::LetterboxInfo& letterbox,
        int orig_width, int orig_height,
        std::vector<DetectedObject>* objects);

    /**
     * @brief 两阶段映射：Letterbox(320) → ROI 局部坐标 → 原始帧坐标
     *        2-stage: Map from letterbox(320) → ROI local → original frame coordinates
     *
     * 适用于先对全图做行人检测，再从行人区域裁剪头部 ROI，在 ROI 上做人脸检测后，
     * 将结果映射回原始全图坐标。适用于远距离低分辨率人脸的检测场景。
     * Used when body detection first crops a head ROI, face detection runs on
     * that ROI, then results are mapped back to the full original frame.
     * Designed for low-resolution faces at long distances.
     */
    static void MapToOriginalFrame(
        const image_utils::LetterboxInfo& letterbox,
        const CropParams& crop,
        int orig_width, int orig_height,
        std::vector<DetectedObject>* objects);

    /**
     * @brief 从人体框自适应裁剪头部 ROI
     *        Adaptive cropping of head ROI from a body bounding box
     *
     * 根据人体高宽比动态选择裁剪策略：
     * - ratio ≥ 2.5 : 全身站立，裁剪上半身并横向扩展
     * - ratio ≥ 1.5 : 半身坐姿，中等裁剪比例
     * - else       : 蹲姿或宽体，较小裁剪比例
     * Dynamically selects cropping strategy based on body aspect ratio.
     */
    static CropParams ComputeHeadROI(
        const Rect& body_bbox,
        int frame_w, int frame_h,
        const Config::HeadROIParams& params);

    /**
     * @brief 坐标归一化到 [0.0, 1.0] 范围
     *        Normalize bbox coordinates to [0.0, 1.0] range
     * @param bbox   待归一化的矩形框 / bbox to normalize
     * @param width  参考宽度（图像宽） / reference width (image width)
     * @param height 参考高度（图像高） / reference height (image height)
     */
    static void NormalizeBBox(Rect& bbox, int width, int height);
};

} // namespace face_rec

#endif // FACE_RECOGNITION_COORDINATE_H
