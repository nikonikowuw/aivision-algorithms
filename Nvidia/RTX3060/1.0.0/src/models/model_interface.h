/**
 * @file model_interface.h
 * @brief GPU Face Recognition — Unified model interface.
 *        Defines abstract interfaces for person detection, face detection,
 *        and face feature extraction models.
 * @module Model Layer
 */

#ifndef GPU_FACE_RECOGNITION_MODEL_INTERFACE_H
#define GPU_FACE_RECOGNITION_MODEL_INTERFACE_H

#include "common/types.h"
#include "runtime/inference_backend.h"
#include <memory>
#include <vector>

namespace face_rec {

/**
 * @class IPersonDetector
 * @brief Abstract interface for person (body) detection model.
 */
class IPersonDetector {
public:
    virtual ~IPersonDetector() = default;

    /**
     * @brief Detect persons in a batch of images.
     * @param images Input images (batch)
     * @param conf_thres Confidence threshold
     * @param nms_iou_thres NMS IoU threshold
     * @return Vector of detected persons per image
     */
    virtual std::vector<std::vector<DetectedObject>> DetectBatch(
        const std::vector<Image>& images,
        float conf_thres, float nms_iou_thres) = 0;
};

/**
 * @class IFaceDetector
 * @brief Abstract interface for face detection model (with landmarks).
 */
class IFaceDetector {
public:
    virtual ~IFaceDetector() = default;

    virtual std::vector<std::vector<DetectedObject>> DetectBatch(
        const std::vector<Image>& images,
        float conf_thres, float nms_iou_thres) = 0;
};

/**
 * @class IFeatureExtractor
 * @brief Abstract interface for face feature extraction model.
 */
class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    /**
     * @brief Extract 512-d embeddings from aligned face images.
     * @param aligned_faces Vector of aligned face images (112×112 BGR)
     * @return Vector of 512-d normalized embedding vectors
     */
    virtual std::vector<std::vector<float>> ExtractBatch(
        const std::vector<std::vector<uint8_t>>& aligned_faces) = 0;
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_MODEL_INTERFACE_H
