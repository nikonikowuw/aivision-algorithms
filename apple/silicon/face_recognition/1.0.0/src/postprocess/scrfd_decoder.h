/**
 * @file scrfd_decoder.h
 * @brief SCRFD 检测器解码器类声明
 *        Declarations for SCRFD detector decoder
 *
 * SCRFD（Sample and Computation Redistribution for Face Detection）是一种
 * 基于 Anchor 的轻量级人脸检测器。该类负责将模型原始输出解码为 BBox 和关键点，
 * 并通过 NMS 去除冗余检测框。
 * SCRFD is an anchor-based lightweight face detector. This class decodes raw
 * model outputs into bounding boxes and landmarks, then applies NMS to remove
 * redundant detections.
 */
#ifndef FACE_RECOGNITION_SCRFD_DECODER_H
#define FACE_RECOGNITION_SCRFD_DECODER_H

#include "common/types.h"
#include <vector>

namespace face_rec {

/**
 * @brief SCRFD 解码器 / SCRFD detection decoder
 *
 * 支持多 stride（如 8/16/32）的特征图解码。每个 stride 对应一个特征层级，
 * 包含分数、边框偏移和关键点偏移输出。
 * Supports multi-stride (e.g. 8/16/32) feature map decoding. Each stride
 * corresponds to a feature level with score, bbox offset, and landmark offset outputs.
 */
class ScrfdDecoder {
public:
    /**
     * @brief 单个 stride 层级的原始模型输出
     *        Raw model output for a single stride level
     */
    struct StrideOutput {
        const float* scores;       // 置信度分数 / confidence scores [grid_h*grid_w*anchor_count]
        const float* bboxes;       // 边框偏移 / bbox offsets [grid_h*grid_w*anchor_count*4]
        const float* kps;          // 关键点偏移 / landmark offsets [grid_h*grid_w*anchor_count*10] (可为 nullptr)
        int stride;                // 下采样步长（如 8/16/32） / downsampling stride
        int anchor_count;          // 每个网格点的 Anchor 数量 / number of anchors per grid cell
        int grid_h;                // 特征图高度 / feature map height
        int grid_w;                // 特征图宽度 / feature map width
    };

    /**
     * @brief 解码基于 Anchor 的 SCRFD 模型输出
     *        Decode anchor-based SCRFD model outputs
     * @param outputs        各 stride 层级的原始输出 / raw outputs per stride level
     * @param conf_threshold 置信度阈值 / confidence threshold
     * @param iou_threshold   NMS 的 IoU 阈值 / NMS IoU threshold
     * @param max_count       最大返回检测数（0 为不限制） / max detections (0=unlimited)
     * @return 解码并过滤后的检测结果列表 / decoded and filtered detection list
     */
    static std::vector<DetectedObject> Decode(
        const std::vector<StrideOutput>& outputs,
        float conf_threshold,
        float iou_threshold,
        int max_count);

    /**
     * @brief 标准非极大值抑制
     *        Standard Non-Maximum Suppression
     * @param detections    待处理的检测结果 / candidate detections
     * @param iou_threshold  IoU 阈值，高于此值的重叠框被抑制 / IoU suppression threshold
     * @param max_count      最大保留检测数（0 为不限制） / max results (0=unlimited)
     * @return NMS 后的检测结果列表 / NMS-filtered detection list
     */
    static std::vector<DetectedObject> NMS(
        const std::vector<DetectedObject>& detections,
        float iou_threshold,
        int max_count);
};

} // namespace face_rec

#endif // FACE_RECOGNITION_SCRFD_DECODER_H
