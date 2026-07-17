/**
 * @file retina_decoder.h
 * @brief GPU Face Recognition — RetinaFace output decoder + NMS.
 *        Decodes raw tensor [batch, N, 15] into face detections with 5 landmarks.
 * @module Postprocessing Layer
 */

#ifndef GPU_FACE_RECOGNITION_RETINA_DECODER_H
#define GPU_FACE_RECOGNITION_RETINA_DECODER_H

#include "common/types.h"
#include <vector>

namespace face_rec {

/**
 * @class RetinaDecoder
 * @brief Decodes RetinaFace raw tensor output into face detections with landmarks.
 */
class RetinaDecoder {
public:
    /**
     * @brief Decode RetinaFace output for a single image.
     * @param raw_output Raw model output (flattened float)
     * @param orig_width Original image width (before resize)
     * @param orig_height Original image height (before resize)
     * @param input_width Network input width (e.g. 640)
     * @param input_height Network input height (e.g. 640)
     * @param conf_thres Confidence threshold
     * @param nms_iou_thres NMS IoU threshold
     * @return Vector of detected face objects with 5 landmarks
     */
    std::vector<DetectedObject> Decode(const std::vector<float>& raw_output,
                                       int orig_width, int orig_height,
                                       int input_width, int input_height,
                                       float conf_thres, float nms_iou_thres);

    /**
     * @brief Batch decode for multiple images.
     */
    std::vector<std::vector<DetectedObject>> DecodeBatch(
        const std::vector<std::vector<float>>& raw_outputs,
        const std::vector<int>& orig_widths, const std::vector<int>& orig_heights,
        int input_width, int input_height,
        float conf_thres, float nms_iou_thres);

private:
    std::vector<DetectedObject> Nms(const std::vector<DetectedObject>& dets,
                                     float iou_threshold);
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_RETINA_DECODER_H
