/**
 * @file adaface_model.h
 * @brief GPU Face Recognition — AdaFace (IR-50) feature extraction model wrapper.
 *        Input: 112×112 aligned face → Output: 512-d L2-normalized embedding.
 * @module Model Layer
 */

#ifndef GPU_FACE_RECOGNITION_ADAFACE_MODEL_H
#define GPU_FACE_RECOGNITION_ADAFACE_MODEL_H

#include "model_interface.h"
#include <memory>

namespace face_rec {

/**
 * @class AdaFaceModel
 * @brief AdaFace IR-50 face feature extraction model.
 */
class AdaFaceModel : public IFeatureExtractor {
public:
    AdaFaceModel();
    ~AdaFaceModel() override;

    bool Initialize(std::shared_ptr<IInferenceBackend> backend);

    std::vector<std::vector<float>> ExtractBatch(
        const std::vector<std::vector<uint8_t>>& aligned_faces) override;

private:
    std::shared_ptr<IInferenceBackend> backend_;

    /**
     * @brief L2-normalize a feature vector in-place.
     */
    static void L2Normalize(std::vector<float>& embedding);
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_ADAFACE_MODEL_H
