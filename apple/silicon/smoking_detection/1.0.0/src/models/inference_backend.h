// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Inference backend contract.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace smoking {

struct ModelSpec {
    std::string model_path;
    std::string model_type;
    int32_t input_width = 640;
    int32_t input_height = 640;
    float conf_threshold = 0.5f;
    float nms_threshold = 0.7f;
};

struct InferenceTensor {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual bool Load(const ModelSpec& spec, std::string& error) = 0;
    virtual bool Run(uint64_t pixel_buffer, InferenceTensor& output,
                     std::string& error) = 0;
    virtual std::string BackendInfo() const = 0;
};

}  // namespace smoking
