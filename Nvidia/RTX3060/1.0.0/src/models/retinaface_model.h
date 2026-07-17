/**
 * @file retinaface_model.h
 * @brief GPU Face Recognition — RetinaFace model wrapper.
 *        Input: 640×640 image → Output: face detections + 5 landmarks.
 * @module Model Layer
 */

#ifndef GPU_FACE_RECOGNITION_RETINAFACE_MODEL_H
#define GPU_FACE_RECOGNITION_RETINAFACE_MODEL_H

#include "model_interface.h"
#include "postprocess/retina_decoder.h"
#include <memory>

namespace face_rec {

/**
 * @class RetinaFaceModel
 * @brief RetinaFace face detection + landmark model.
 */
class RetinaFaceModel : public IFaceDetector {
public:
    RetinaFaceModel();
    ~RetinaFaceModel() override;

    bool Initialize(std::shared_ptr<IInferenceBackend> backend, int input_size = 640);

    std::vector<std::vector<DetectedObject>> DetectBatch(
        const std::vector<Image>& images,
        float conf_thres, float nms_iou_thres) override;

private:
    std::shared_ptr<IInferenceBackend> backend_;
    std::unique_ptr<RetinaDecoder> decoder_;
    int input_size_ = 640;
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_RETINAFACE_MODEL_H
