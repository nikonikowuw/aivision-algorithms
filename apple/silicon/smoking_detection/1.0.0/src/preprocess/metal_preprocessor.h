// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Metal and BGR24 preprocessing.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "../common/types.h"

namespace smoking {

struct PreparedImage {
    uint64_t pixel_buffer = 0;  // Borrowed until the next Process call.
    ImageTransform transform;
};

class MetalPreprocessor {
public:
    MetalPreprocessor();
    ~MetalPreprocessor();

    bool Initialize(int32_t model_width, int32_t model_height,
                    const std::string& metallib_path, std::string& error);
    void Destroy();

    bool ProcessNative(uint64_t cv_pixel_buffer, int32_t frame_width,
                       int32_t frame_height, uint32_t pixel_format,
                       const RectF& roi, PreparedImage& prepared,
                       std::string& error);

    bool ProcessBGR24(const uint8_t* frame, int32_t frame_width,
                      int32_t frame_height, int32_t stride,
                      const RectF& roi, PreparedImage& prepared,
                      std::string& error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace smoking
