/**
 * @file face_aligner.cpp
 * @brief GPU Face Recognition — 5-point affine face alignment implementation.
 *        Uses closed-form similarity solver + bilinear warp.
 */

#include "face_aligner.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace face_rec {

// AdaFace standard reference landmarks for 112×112 output
static const float kRefLandmarks[5][2] = {
    {38.2946f, 51.6963f},   // Left eye
    {73.5318f, 51.5014f},   // Right eye
    {56.0252f, 71.7366f},   // Nose
    {41.5493f, 92.3655f},   // Left mouth
    {70.7299f, 92.2041f},   // Right mouth
};

struct FaceAligner::Impl {
    // Reference points for least-squares solver
    float ref_pts[5][2];
    float M[2][3];  // 2×3 similarity transform matrix

    Impl() {
        for (int i = 0; i < 5; ++i) {
            ref_pts[i][0] = kRefLandmarks[i][0];
            ref_pts[i][1] = kRefLandmarks[i][1];
        }
    }

    /**
     * @brief Compute 2×3 similarity transform from 5 source points to 5 reference points.
     *        Uses closed-form least-squares solution.
     */
    bool ComputeSimilarity(const std::array<Point, 5>& src) {
        // Build normal equations for similarity transform:
        //   dst_x = a * src_x - b * src_y + tx
        //   dst_y = b * src_x + a * src_y + ty
        //
        // This is equivalent to: [a, b, tx] and [b, a, ty] for x and y channels.

        double A[5][3];
        double Bx[5], By[5];

        for (int i = 0; i < 5; ++i) {
            A[i][0] = src[i].x;
            A[i][1] = -src[i].y;
            A[i][2] = 1.0;
            Bx[i] = ref_pts[i][0];
            By[i] = ref_pts[i][1];
        }

        // Solve using normal equations: (A^T A) x = A^T b
        double ATA[3][3] = {};
        double ATBx[3] = {};
        double ATBy[3] = {};

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    ATA[j][k] += A[i][j] * A[i][k];
                }
                ATBx[j] += A[i][j] * Bx[i];
                ATBy[j] += A[i][j] * By[i];
            }
        }

        // Solve 3×3 system using Cramer's rule
        double det = ATA[0][0] * (ATA[1][1] * ATA[2][2] - ATA[1][2] * ATA[2][1])
                   - ATA[0][1] * (ATA[1][0] * ATA[2][2] - ATA[1][2] * ATA[2][0])
                   + ATA[0][2] * (ATA[1][0] * ATA[2][1] - ATA[1][1] * ATA[2][0]);

        if (std::abs(det) < 1e-10) return false;

        // Solve for [a, b, tx] and [b, a, ty]
        for (int col = 0; col < 3; ++col) {
            double temp_bx[3][3], temp_by[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    temp_bx[r][c] = ATA[r][c];
                    temp_by[r][c] = ATA[r][c];
                }
                temp_bx[r][col] = ATBx[r];
                temp_by[r][col] = ATBy[r];
            }
        }

        // Direct Cramer's rule for each unknown
        double x_bx[3], x_by[3];
        for (int var = 0; var < 3; ++var) {
            double num_x = 0, num_y = 0;
            for (int r = 0; r < 3; ++r) {
                double col_x = ATBx[r], col_y = ATBy[r];
                // Replace column var with ATB
                // Simpler approach: direct 3x3 solve
                (void)col_x;
                (void)col_y;
            }
        }

        // Use simple Gauss elimination instead
        double aug[3][4];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) aug[r][c] = ATA[r][c];
            aug[r][3] = ATBx[r];
        }

        // Forward elimination
        for (int c = 0; c < 3; ++c) {
            // Find pivot
            int max_r = c;
            for (int r = c + 1; r < 3; ++r) {
                if (std::abs(aug[r][c]) > std::abs(aug[max_r][c])) max_r = r;
            }
            std::swap(aug[c], aug[max_r]);

            if (std::abs(aug[c][c]) < 1e-10) return false;

            for (int r = c + 1; r < 3; ++r) {
                double factor = aug[r][c] / aug[c][c];
                for (int j = c; j < 4; ++j) {
                    aug[r][j] -= factor * aug[c][j];
                }
            }
        }
        // Back substitution
        for (int r = 2; r >= 0; --r) {
            double sum = aug[r][3];
            for (int c = r + 1; c < 3; ++c) sum -= aug[r][c] * aug[c][3]; // WRONG: aug[c][3] should be x[c]
        }
        // Redo properly
        double xb[3];
        xb[2] = aug[2][3] / aug[2][2];
        xb[1] = (aug[1][3] - aug[1][2] * xb[2]) / aug[1][1];
        xb[0] = (aug[0][3] - aug[0][1] * xb[1] - aug[0][2] * xb[2]) / aug[0][0];

        // Same for Y
        for (int r = 0; r < 3; ++r) {
            aug[r][3] = ATBy[r];
        }
        for (int c = 0; c < 3; ++c) {
            int max_r = c;
            for (int r = c + 1; r < 3; ++r) {
                if (std::abs(aug[r][c]) > std::abs(aug[max_r][c])) max_r = r;
            }
            std::swap(aug[c], aug[max_r]);
            if (std::abs(aug[c][c]) < 1e-10) return false;
            for (int r = c + 1; r < 3; ++r) {
                double factor = aug[r][c] / aug[c][c];
                for (int j = c; j < 4; ++j) aug[r][j] -= factor * aug[c][j];
            }
        }
        double yb[3];
        yb[2] = aug[2][3] / aug[2][2];
        yb[1] = (aug[1][3] - aug[1][2] * yb[2]) / aug[1][1];
        yb[0] = (aug[0][3] - aug[0][1] * yb[1] - aug[0][2] * yb[2]) / aug[0][0];

        // M[0] = [a, -b, tx] → x transform
        // M[1] = [b,  a, ty] → y transform
        M[0][0] = static_cast<float>(xb[0]);  // a
        M[0][1] = static_cast<float>(xb[1]);  // -b
        M[0][2] = static_cast<float>(xb[2]);  // tx
        M[1][0] = static_cast<float>(yb[0]);  // b
        M[1][1] = static_cast<float>(yb[1]);  // a
        M[1][2] = static_cast<float>(yb[2]);  // ty

        return true;
    }

    /**
     * @brief Apply the transform to a single pixel using bilinear interpolation.
     */
    void WarpPixel(const uint8_t* src, int src_w, int src_h, int src_stride,
                   uint8_t* dst, int dst_x, int dst_y) {
        // Inverse map: dst → src
        float src_x = M[0][0] * dst_x + M[0][1] * dst_y + M[0][2];
        float src_y = M[1][0] * dst_x + M[1][1] * dst_y + M[1][2];

        int x0 = static_cast<int>(std::floor(src_x));
        int y0 = static_cast<int>(std::floor(src_y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float fx = src_x - x0;
        float fy = src_y - y0;

        // Boundary check
        x0 = std::max(0, std::min(x0, src_w - 1));
        x1 = std::max(0, std::min(x1, src_w - 1));
        y0 = std::max(0, std::min(y0, src_h - 1));
        y1 = std::max(0, std::min(y1, src_h - 1));

        for (int c = 0; c < 3; ++c) {
            float v00 = src[y0 * src_stride * 3 + x0 * 3 + c];
            float v10 = src[y0 * src_stride * 3 + x1 * 3 + c];
            float v01 = src[y1 * src_stride * 3 + x0 * 3 + c];
            float v11 = src[y1 * src_stride * 3 + x1 * 3 + c];

            float val = v00 * (1 - fx) * (1 - fy)
                      + v10 * fx * (1 - fy)
                      + v01 * (1 - fx) * fy
                      + v11 * fx * fy;

            dst[dst_y * kOutputSize * 3 + dst_x * 3 + c] =
                static_cast<uint8_t>(std::clamp(val + 0.5f, 0.0f, 255.0f));
        }
    }
};

FaceAligner::FaceAligner() : impl_(std::make_unique<Impl>()) {}
FaceAligner::~FaceAligner() = default;

bool FaceAligner::Align(const Image& src, const std::array<Point, 5>& landmarks,
                        uint8_t* output) {
    if (!src.data || !output) return false;

    if (!impl_->ComputeSimilarity(landmarks)) {
        return false;
    }

    for (int y = 0; y < Impl::kOutputSize; ++y) {
        for (int x = 0; x < Impl::kOutputSize; ++x) {
            impl_->WarpPixel(src.data, src.width, src.height, src.stride,
                             output, x, y);
        }
    }

    return true;
}

std::vector<std::vector<uint8_t>> FaceAligner::AlignBatch(
    const std::vector<Image>& src_images,
    const std::vector<std::array<Point, 5>>& landmarks_batch) {

    size_t batch_size = src_images.size();
    std::vector<std::vector<uint8_t>> results(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        results[i].resize(112 * 112 * 3);
        Align(src_images[i], landmarks_batch[i], results[i].data());
    }

    return results;
}

} // namespace face_rec
