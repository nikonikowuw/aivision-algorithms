// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Native CoreML backend.

#pragma once

#include <memory>

#include "../models/inference_backend.h"

namespace smoking {

class CoreMLNativeBackend final : public IInferenceBackend {
public:
    CoreMLNativeBackend();
    ~CoreMLNativeBackend() override;

    bool Load(const ModelSpec& spec, std::string& error) override;
    bool Run(uint64_t pixel_buffer, InferenceTensor& output,
             std::string& error) override;
    std::string BackendInfo() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace smoking
