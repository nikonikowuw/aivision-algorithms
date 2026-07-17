// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - DAMO-YOLO detector wrapper.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../common/types.h"
#include "inference_backend.h"

namespace smoking {

class DamoYoloDetector {
public:
    explicit DamoYoloDetector(std::unique_ptr<IInferenceBackend> backend);
    ~DamoYoloDetector();

    bool Load(const ModelSpec& spec, std::string& error);
    ErrorCode Detect(uint64_t prepared_pixel_buffer,
                     const ImageTransform& transform,
                     std::vector<Detection>& detections,
                     std::string& error);
    void Unload();

private:
    std::unique_ptr<IInferenceBackend> backend_;
    ModelSpec spec_;
    bool loaded_ = false;
};

}  // namespace smoking
