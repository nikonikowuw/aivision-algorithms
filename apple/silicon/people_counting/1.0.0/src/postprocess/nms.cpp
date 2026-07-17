/**
 * @file nms.cpp
 * @brief YOLOv8 decoder and NMS implementation
 */

#include "nms.h"
#include "common/geometry_utils.h"
#include "common/logger.h"
#include <algorithm>

namespace people_count {

std::vector<DetectedObject> DecodeYOLOv8(const float* output_buffer,
                                         const int64_t shape[4],
                                         float conf_threshold,
                                         float iou_threshold,
                                         int img_w, int img_h) {
    std::vector<DetectedObject> detections;
    if (!output_buffer) return detections;

    // Detect format from shape
    // standard output shape from trt/coreml: [1, 84, 8400] or [1, 8400, 84] (or [1, 5, 8400] for person only)
    int64_t d1 = shape[1];
    int64_t d2 = shape[2];

    int num_anchors = 0;
    int num_channels = 0;
    bool transposed = false;

    if (d1 < d2) {
        num_channels = static_cast<int>(d1);
        num_anchors = static_cast<int>(d2);
        transposed = false;
    } else {
        num_anchors = static_cast<int>(d1);
        num_channels = static_cast<int>(d2);
        transposed = true;
    }

    if (num_anchors <= 0 || num_channels < 5) {
        ALGO_LOGE(DETECTOR, "Invalid output shape: [%lld, %lld, %lld, %lld]",
                  shape[0], shape[1], shape[2], shape[3]);
        return detections;
    }

    ALGO_LOGD(DETECTOR, "YOLOv8 output parsed: anchors=%d, channels=%d, transposed=%s",
              num_anchors, num_channels, transposed ? "true" : "false");

    for (int i = 0; i < num_anchors; ++i) {
        float cx, cy, w, h, conf;
        if (!transposed) {
            cx = output_buffer[0 * num_anchors + i];
            cy = output_buffer[1 * num_anchors + i];
            w  = output_buffer[2 * num_anchors + i];
            h  = output_buffer[3 * num_anchors + i];
            conf = output_buffer[4 * num_anchors + i]; // class 0 (person)
        } else {
            cx = output_buffer[i * num_channels + 0];
            cy = output_buffer[i * num_channels + 1];
            w  = output_buffer[i * num_channels + 2];
            h  = output_buffer[i * num_channels + 3];
            conf = output_buffer[i * num_channels + 4]; // class 0 (person)
        }

        if (conf < conf_threshold) continue;

        // Convert [cx, cy, w, h] to [x, y, w, h] (top-left)
        float x = cx - w / 2.0f;
        float y = cy - h / 2.0f;

        // Clip coordinates to network dimensions
        x = std::max(0.0f, std::min(x, static_cast<float>(img_w)));
        y = std::max(0.0f, std::min(y, static_cast<float>(img_h)));
        float width = std::max(0.0f, std::min(w, static_cast<float>(img_w - x)));
        float height = std::max(0.0f, std::min(h, static_cast<float>(img_h - y)));

        DetectedObject obj;
        obj.bbox = Rect{x, y, width, height};
        obj.confidence = conf;
        obj.label = CAT_PERSON;
        obj.track_id = -1;
        obj.crossed = false;
        obj.direction = "none";

        detections.push_back(obj);
    }

    ALGO_LOGD(DETECTOR, "Decoded %zu candidates before NMS", detections.size());
    return ApplyNMS(detections, iou_threshold);
}

std::vector<DetectedObject> ApplyNMS(const std::vector<DetectedObject>& detections,
                                     float iou_threshold) {
    std::vector<DetectedObject> results;
    if (detections.empty()) return results;

    std::vector<int> indices(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        indices[i] = static_cast<int>(i);
    }

    std::sort(indices.begin(), indices.end(), [&detections](int i1, int i2) {
        return detections[i1].confidence > detections[i2].confidence;
    });

    std::vector<bool> active(detections.size(), true);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (!active[idx]) continue;

        results.push_back(detections[idx]);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            int target_idx = indices[j];
            if (!active[target_idx]) continue;

            float iou = ComputeIoU(detections[idx].bbox, detections[target_idx].bbox);
            if (iou >= iou_threshold) {
                active[target_idx] = false;
            }
        }
    }

    ALGO_LOGD(DETECTOR, "NMS reduced detections to %zu objects", results.size());
    return results;
}

} // namespace people_count
