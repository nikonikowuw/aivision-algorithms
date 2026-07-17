/**
 * @file trt_backend.cpp
 * @brief GPU Face Recognition — TensorRT backend implementation.
 *        Loads serialized .engine files, manages CUDA streams and device memory,
 *        supports dynamic batch (min=1, opt=8, max=16).
 */

#include "trt_backend.h"
#include "cuda_utils.h"
#include "common/logger.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <cassert>

namespace face_rec {

// ---- TensorRT Logger ----

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                ALGO_LOGE(BACKEND, "TRT INTERNAL: %s", msg);
                break;
            case Severity::kERROR:
                ALGO_LOGE(BACKEND, "TRT ERROR: %s", msg);
                break;
            case Severity::kWARNING:
                ALGO_LOGW(BACKEND, "TRT WARN: %s", msg);
                break;
            case Severity::kINFO:
                ALGO_LOGI(BACKEND, "TRT INFO: %s", msg);
                break;
            default:
                ALGO_LOGD(BACKEND, "TRT VERBOSE: %s", msg);
                break;
        }
    }
};

static TrtLogger s_trt_logger;

// ---- Helper: Read engine file into buffer ----

static std::vector<char> ReadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        ALGO_LOGE(BACKEND, "Failed to open engine file: %s", path.c_str());
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        ALGO_LOGE(BACKEND, "Failed to read engine file: %s", path.c_str());
        return {};
    }
    return buffer;
}

// ---- TrtBackend::Impl ----

struct TrtBackend::Impl {
    int cuda_device = 0;
    size_t max_workspace = 2ULL * 1024 * 1024 * 1024;

    // TensorRT runtime and engine
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1.ICudaEngine> engine;
    std::unique_ptr<nvinfer1.IExecutionContext> context;

    // CUDA stream
    CudaStream stream;

    // Device buffers for input/output bindings
    std::vector<CudaDeviceBuffer> device_buffers;
    std::vector<size_t> binding_sizes;
    std::vector<nvinfer1::Dims> binding_dims;
    std::vector<bool> binding_is_input;

    // Host output buffers
    std::vector<std::vector<float>> host_outputs;

    int num_bindings = 0;
    int current_batch_size = 1;

    bool LoadEngine(const std::string& path) {
        auto data = ReadEngineFile(path);
        if (data.empty()) return false;

        runtime.reset(nvinfer1::createInferRuntime(s_trt_logger));
        if (!runtime) {
            ALGO_LOGE(BACKEND, "Failed to create TensorRT runtime");
            return false;
        }

        engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
        if (!engine) {
            ALGO_LOGE(BACKEND, "Failed to deserialize CUDA engine from: %s", path.c_str());
            return false;
        }

        context.reset(engine->createExecutionContext());
        if (!context) {
            ALGO_LOGE(BACKEND, "Failed to create execution context");
            return false;
        }

        num_bindings = engine->getNbBindings();
        binding_sizes.resize(num_bindings);
        binding_dims.resize(num_bindings);
        binding_is_input.resize(num_bindings);
        device_buffers.resize(num_bindings);
        host_outputs.resize(num_bindings);

        ALGO_LOGI(BACKEND, "TensorRT engine loaded: %s (%d bindings)", path.c_str(), num_bindings);

        for (int i = 0; i < num_bindings; ++i) {
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            binding_dims[i] = dims;
            binding_is_input[i] = engine->bindingIsInput(i);

            size_t elem_count = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                // Replace -1 (dynamic) with current batch size
                int dim = (dims.d[d] == -1) ? current_batch_size : dims.d[d];
                elem_count *= dim;
            }
            binding_sizes[i] = elem_count * sizeof(float);

            ALGO_LOGI(BACKEND, "  Binding[%d] %s: %s, elements=%zu",
                      i, binding_is_input[i] ? "INPUT" : "OUTPUT",
                      engine->getBindingName(i), elem_count);
        }

        return true;
    }

    void AllocateBuffers() {
        for (int i = 0; i < num_bindings; ++i) {
            if (!device_buffers[i].Allocate(binding_sizes[i])) {
                ALGO_LOGE(BACKEND, "Failed to allocate device buffer for binding %d", i);
            }
            if (!binding_is_input[i]) {
                host_outputs[i].resize(binding_sizes[i] / sizeof(float));
            }
        }
    }

    void SetBindingDimensions(int batch_size) {
        current_batch_size = batch_size;
        for (int i = 0; i < num_bindings; ++i) {
            nvinfer1::Dims dims = binding_dims[i];
            // Update batch dimension (first dim) for dynamic shapes
            if (dims.d[0] == -1) {
                dims.d[0] = batch_size;
                context->setBindingDimensions(i, dims);
            }
            // Recompute binding size
            size_t elem_count = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                elem_count *= dims.d[d];
            }
            binding_sizes[i] = elem_count * sizeof(float);
            if (!binding_is_input[i]) {
                host_outputs[i].resize(binding_sizes[i] / sizeof(float));
            }
        }
    }
};

// ---- TrtBackend Public API ----

TrtBackend::TrtBackend(int cuda_device, size_t max_workspace)
    : cuda_device_(cuda_device), max_workspace_(max_workspace) {}

TrtBackend::~TrtBackend() {
    Unload();
}

bool TrtBackend::Load(const std::string& engine_path) {
    impl_ = std::make_unique<Impl>();
    impl_->cuda_device = cuda_device_;
    impl_->max_workspace = max_workspace_;

    if (!CudaSetDevice(cuda_device_)) {
        return false;
    }

    if (!impl_->LoadEngine(engine_path)) {
        return false;
    }

    impl_->AllocateBuffers();

    ALGO_LOGI(BACKEND, "TrtBackend loaded successfully on CUDA device %d", cuda_device_);
    return true;
}

bool TrtBackend::Run(const std::vector<ModelInput>& inputs,
                     std::vector<ModelOutput>* outputs) {
    if (!impl_ || !impl_->context) {
        ALGO_LOGE(BACKEND, "TrtBackend::Run called but backend not loaded");
        return false;
    }

    // Determine batch size from first input
    int batch_size = 1;
    if (!inputs.empty() && inputs[0].shape_len > 0) {
        batch_size = static_cast<int>(inputs[0].shape[0]);
    }

    // Update dynamic shapes for this batch
    impl_->SetBindingDimensions(batch_size);

    // Re-allocate device buffers if sizes changed
    impl_->AllocateBuffers();

    // Copy inputs to device
    for (size_t i = 0; i < inputs.size() && i < static_cast<size_t>(impl_->num_bindings); ++i) {
        if (!impl_->binding_is_input[i]) continue;
        if (!CudaMemcpyH2D(impl_->device_buffers[i].Get(), inputs[i].data,
                           inputs[i].size * sizeof(float), impl_->stream.Get())) {
            ALGO_LOGE(BACKEND, "H2D copy failed for input %zu", i);
            return false;
        }
    }

    // Execute inference
    std::vector<void*> device_ptrs(impl_->num_bindings);
    for (int i = 0; i < impl_->num_bindings; ++i) {
        device_ptrs[i] = impl_->device_buffers[i].Get();
    }

    bool status = impl_->context->enqueueV2(device_ptrs.data(), impl_->stream.Get(), nullptr);
    if (!status) {
        ALGO_LOGE(BACKEND, "TensorRT enqueueV2 failed");
        return false;
    }

    // Copy outputs from device to host
    if (outputs) {
        outputs->clear();
        for (int i = 0; i < impl_->num_bindings; ++i) {
            if (impl_->binding_is_input[i]) continue;

            if (!CudaMemcpyD2H(impl_->host_outputs[i].data(),
                               impl_->device_buffers[i].Get(),
                               impl_->binding_sizes[i],
                               impl_->stream.Get())) {
                ALGO_LOGE(BACKEND, "D2H copy failed for output %d", i);
                return false;
            }

            ModelOutput out;
            out.name = impl_->engine->getBindingName(i);
            out.buffer = impl_->host_outputs[i];

            nvinfer1::Dims dims = impl_->context->getBindingDimensions(i);
            for (int d = 0; d < dims.nbDims && d < 4; ++d) {
                out.shape[d] = dims.d[d];
            }
            outputs->push_back(std::move(out));
        }
    }

    // Synchronize to ensure completion
    if (!impl_->stream.Synchronize()) {
        return false;
    }

    return true;
}

void TrtBackend::Unload() {
    if (impl_) {
        impl_->stream.Synchronize();
        for (auto& buf : impl_->device_buffers) {
            buf.Free();
        }
        impl_->context.reset();
        impl_->engine.reset();
        impl_->runtime.reset();
        impl_.reset();
    }
}

int TrtBackend::GetNumInputs() const {
    if (!impl_) return 0;
    int count = 0;
    for (int i = 0; i < impl_->num_bindings; ++i) {
        if (impl_->binding_is_input[i]) ++count;
    }
    return count;
}

int TrtBackend::GetNumOutputs() const {
    if (!impl_) return 0;
    int count = 0;
    for (int i = 0; i < impl_->num_bindings; ++i) {
        if (!impl_->binding_is_input[i]) ++count;
    }
    return count;
}

std::vector<int64_t> TrtBackend::GetInputShape(int index) const {
    if (!impl_) return {};
    int input_idx = 0;
    for (int i = 0; i < impl_->num_bindings; ++i) {
        if (impl_->binding_is_input[i]) {
            if (input_idx == index) {
                std::vector<int64_t> shape;
                for (int d = 0; d < impl_->binding_dims[i].nbDims; ++d) {
                    shape.push_back(impl_->binding_dims[i].d[d]);
                }
                return shape;
            }
            ++input_idx;
        }
    }
    return {};
}

std::vector<int64_t> TrtBackend::GetOutputShape(int index) const {
    if (!impl_) return {};
    int output_idx = 0;
    for (int i = 0; i < impl_->num_bindings; ++i) {
        if (!impl_->binding_is_input[i]) {
            if (output_idx == index) {
                std::vector<int64_t> shape;
                nvinfer1::Dims dims = impl_->context->getBindingDimensions(i);
                for (int d = 0; d < dims.nbDims; ++d) {
                    shape.push_back(dims.d[d]);
                }
                return shape;
            }
            ++output_idx;
        }
    }
    return {};
}

void TrtBackend::SetBatchSize(int batch_size) {
    if (impl_) {
        impl_->current_batch_size = batch_size;
    }
}

} // namespace face_rec
