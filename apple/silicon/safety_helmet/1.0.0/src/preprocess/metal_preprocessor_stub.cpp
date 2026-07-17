/**
 * @file metal_preprocessor_stub.cpp
 * @brief Non-Apple no-op stub for MetalPreprocessor.
 *
 * On platforms without Metal (Linux, Windows, etc.), MetalPreprocessor is
 * still compiled into the library so the pipeline code doesn't need
 * #ifdef guards. All methods return false/not-available, and the pipeline
 * transparently falls back to CPU-based LetterBoxPreprocess().
 */

#include "metal_preprocessor.h"

namespace safety_helmet {

// Stub pimpl — no fields needed.
struct MetalPreprocessor::Impl {};

MetalPreprocessor::MetalPreprocessor() : impl_(std::make_unique<Impl>()) {}
MetalPreprocessor::~MetalPreprocessor() = default;

bool MetalPreprocessor::IsAvailable() const { return false; }

bool MetalPreprocessor::Process(const cv::Mat&,
                                std::vector<float>&,
                                int,
                                LetterBoxInfo&) {
    return false;
}

bool MetalPreprocessor::ProcessFromPixelBuffer(void*,
                                                std::vector<float>&,
                                                int,
                                                LetterBoxInfo&) {
    return false;
}

} // namespace safety_helmet
