/**
 * @file yolo_decoder.h
 * @brief GPU Face Recognition — YOLO11n output decoder + NMS.
 * @module Postprocessing Layer
 */

#ifndef GPU_FACE_RECOGNITION_YOLO_DECODER_H
#define GPU_FACE_RECOGNITION_YOLO_DECODER_H

#include "common/types.h"
#include <vector>

namespace face_rec {

/**
 * @class YoloDecoder
 * @brief Decodes YOLO11n raw tensor output [batch, 8400, 84] into detected objects.
 *        Applies score thresholding and CPU greedy NMS.
 */
class YoloDecoder {
public:
    /**
     * @brief Decode YOLO11n output for a single image.
     * @param raw_output Raw model output tensor (flattened float)
     * @param orig_width Original image width (before letterbox)
     * @param orig_height Original image height (before letterbox)
     * @param scale Letterbox scale factor
     * @param pad_x Letterbox X padding
     * @param pad_y Letterbox Y padding
     * @param conf_thres Confidence threshold
     * @param nms_iou_thres NMS IoU threshold
     * @return Vector of detected person bounding boxes in original coordinates
     */
    std::vector<DetectedObject> Decode(const std::vector<float>& raw_output,
                                       int orig_width, int orig_height,
                                       float scale, int pad_x, int pad_y,
                                       float conf_thres, float nms_iou_thres);

    /**
     * @brief Batch decode: decode outputs for multiple images.
     */
    std::vector<std::vector<DetectedObject>> DecodeBatch(
        const std::vector<std::vector<float>>& raw_outputs,
        const std::vector<int>& orig_widths, const std::vector<int>& orig_heights,
        const std::vector<float>& scales,
        const std::vector<int>& pad_xs, const std::vector<int>& pad_ys,
        float conf_thres, float nms_iou_thres);

private:
    std::vector<DetectedObject> Nms(const std::vector<DetectedObject>& dets,
                                     float iou_threshold);
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_YOLO_DECODER_H
