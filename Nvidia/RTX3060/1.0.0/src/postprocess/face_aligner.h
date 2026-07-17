/**
 * @file face_aligner.h
 * @brief GPU Face Recognition — 5-point affine face alignment.
 *        Transforms face to 112×112 using standard AdaFace reference landmarks.
 * @module Postprocessing Layer
 */

#ifndef GPU_FACE_RECOGNITION_FACE_ALIGNER_H
#define GPU_FACE_RECOGNITION_FACE_ALIGNER_H

#include "common/types.h"
#include <vector>
#include <memory>

namespace face_rec {

/**
 * @class IFaceAligner
 * @brief Abstract interface for face alignment.
 */
class IFaceAligner {
public:
    virtual ~IFaceAligner() = default;

    /**
     * @brief Align a single face to 112×112 using 5-point affine transform.
     * @param src Source image
     * @param landmarks 5 face landmarks
     * @param output Aligned 112×112 BGR image (caller pre-allocates 112*112*3 bytes)
     * @return true on success
     */
    virtual bool Align(const Image& src, const std::array<Point, 5>& landmarks,
                       uint8_t* output) = 0;

    /**
     * @brief Align a batch of faces.
     */
    virtual std::vector<std::vector<uint8_t>> AlignBatch(
        const std::vector<Image>& src_images,
        const std::vector<std::array<Point, 5>>& landmarks_batch) = 0;
};

/**
 * @class FaceAligner
 * @brief CPU-based 5-point affine alignment using closed-form similarity solver.
 *        Uses bilinear warp for sub-pixel accuracy.
 */
class FaceAligner : public IFaceAligner {
public:
    FaceAligner();
    ~FaceAligner() override;

    bool Align(const Image& src, const std::array<Point, 5>& landmarks,
               uint8_t* output) override;

    std::vector<std::vector<uint8_t>> AlignBatch(
        const std::vector<Image>& src_images,
        const std::vector<std::array<Point, 5>>& landmarks_batch) override;

    /** @brief AdaFace standard reference landmarks for 112×112 output */
    static constexpr int kOutputSize = 112;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_FACE_ALIGNER_H
