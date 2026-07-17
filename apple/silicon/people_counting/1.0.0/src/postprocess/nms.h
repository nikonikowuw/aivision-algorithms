/**
 * @file nms.h
 * @brief YOLO postprocessing and NMS
 */

#ifndef PEOPLE_COUNTING_NMS_H
#define PEOPLE_COUNTING_NMS_H

#include "common/types.h"
#include <vector>

namespace people_count {

/**
 * @brief Decode YOLOv8 outputs and apply NMS to filter detections
 * @param output_buffer The raw flat float array from CoreML output
 * @param shape The shape of the output tensor (4 dimensions)
 * @param conf_threshold Confidence score threshold
 * @param iou_threshold NMS IoU threshold
 * @param img_w Input image width (typically 640)
 * @param img_h Input image height (typically 640)
 * @return Detections inside the 640x640 coordinate space
 */
std::vector<DetectedObject> DecodeYOLOv8(const float* output_buffer,
                                         const int64_t shape[4],
                                         float conf_threshold,
                                         float iou_threshold,
                                         int img_w, int img_h);

std::vector<DetectedObject> ApplyNMS(const std::vector<DetectedObject>& detections,
                                     float iou_threshold);

} // namespace people_count

#endif // PEOPLE_COUNTING_NMS_H
