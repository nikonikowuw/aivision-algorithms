/**
 * @file inference_backend.cpp
 * @brief ONNX Runtime 后端实现：Session 创建、执行提供者选择（XNNPACK/CoreML/CPU）、动态输入输出处理
 *        ONNX Runtime backend implementation: Session creation, execution provider selection
 *        (XNNPACK/CoreML/CPU), dynamic input/output handling
 */
#include "inference_backend.h"
#include "common/logger.h"
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#include <cpu_provider_factory.h>
#endif
#include <cstring>
#include <algorithm>

namespace face_rec {

/**
 * @struct OnnxBackend::Impl
 * @brief OnnxBackend 的内部实现（PIMPL 模式）
 *        Internal implementation of OnnxBackend (PIMPL pattern)
 *
 * 封装 Ort::Env、Ort::Session 以及输入输出节点名称，
 * 避免在头文件中暴露 ONNX Runtime 类型。
 * Encapsulates Ort::Env, Ort::Session, and input/output node names,
 * avoiding exposure of ONNX Runtime types in the header.
 */
struct OnnxBackend::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "OnnxBackend"};  ///< ONNX Runtime 环境（日志级别 WARNING）/ ORT environment
    std::unique_ptr<Ort::Session> session;   ///< 推理会话 / inference session
    std::vector<std::string> input_node_names;   ///< 输入节点名称 / input node names
    std::vector<std::string> output_node_names;  ///< 输出节点名称 / output node names
    std::vector<const char*> input_names_char;   ///< 输入名称 C 字符串指针 / input name C-string pointers
    std::vector<const char*> output_names_char;  ///< 输出名称 C 字符串指针 / output name C-string pointers
    Ort::MemoryInfo memory_info{nullptr};        ///< 内存信息（CPU 分配器）/ memory info (CPU allocator)
    
    Impl() {
        // 创建 CPU 内存信息供张量绑定使用
        // Create CPU memory info for tensor binding
        memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    }
};

OnnxBackend::OnnxBackend(OrtProvider provider, int intra_op_threads)
    : provider_(provider), intra_op_threads_(intra_op_threads), impl_(std::make_unique<Impl>()) {}

OnnxBackend::~OnnxBackend() = default;

void OnnxBackend::Unload() {
    // 释放 Session 并清空节点名称缓存
    // Release Session and clear node name caches
    impl_->session.reset();
    impl_->input_node_names.clear();
    impl_->output_node_names.clear();
    impl_->input_names_char.clear();
    impl_->output_names_char.clear();
}

bool OnnxBackend::Load(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(intra_op_threads_);   ///< 算子内并行线程数 / intra-op thread count
        session_options.SetInterOpNumThreads(1);                    ///< 算子间并行线程数（固定 1）/ inter-op thread count (fixed to 1)
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);  ///< 启用所有图优化 / enable all graph optimizations
        session_options.EnableCpuMemArena();  ///< 启用 CPU 内存池（减少分配开销）/ enable CPU memory arena (reduce allocation overhead)
        
        std::string provider_name = "CPU";
        
        if (provider_ == OrtProvider::COREML) {
            // ---- Apple CoreML Execution Provider ----
            // CoreML EP 可将算子委托给 ANE（神经网络引擎）+ GPU + CPU 混合执行
            // CoreML EP can delegate ops to ANE (Neural Engine) + GPU + CPU hybrid execution
#ifdef __APPLE__
            uint32_t coreml_flags = COREML_FLAG_USE_NONE;
            coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;  ///< 允许子图级别委派 / allow subgraph-level delegation
            
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags);
            if (status != nullptr) {
                Ort::GetApi().ReleaseStatus(status);
                ALGO_LOGW(CONFIG, "Failed to enable CoreML EP, falling back to CPU.");
            } else {
                provider_name = "CoreML";
                ALGO_LOGI(CONFIG, "CoreML Execution Provider successfully enabled.");
            }
#else
            ALGO_LOGW(CONFIG, "CoreML is only supported on Apple platforms. Falling back to CPU.");
#endif
        } else if (provider_ == OrtProvider::XNNPACK) {
            // ---- XNNPACK Execution Provider ----
            // XNNPACK 是针对 ARM CPU 优化的低精度推理库（支持 FP32/FP16/QNNPACK 子集）
            // XNNPACK is a low-precision inference library optimized for ARM CPUs
            session_options.AddConfigEntry("session.use_xnnpack", "1");
            provider_name = "XNNPACK";
            ALGO_LOGI(CONFIG, "Enabled XNNPACK optimization.");
        }
        
        ALGO_LOGI(CONFIG, "Loading ONNX model: %s using EP: %s", model_path.c_str(), provider_name.c_str());
        
        // 创建 Session：Windows 使用宽字符串，其他平台使用 UTF-8
        // Create Session: Windows uses wide string, other platforms use UTF-8
#ifdef _WIN32
        std::wstring w_model_path(model_path.begin(), model_path.end());
        impl_->session = std::make_unique<Ort::Session>(impl_->env, w_model_path.c_str(), session_options);
#else
        impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), session_options);
#endif
        
        // 获取输入节点名称
        // Query input node names
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = impl_->session->GetInputCount();
        impl_->input_node_names.resize(num_inputs);
        impl_->input_names_char.resize(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto allocated_name = impl_->session->GetInputNameAllocated(i, allocator);
            impl_->input_node_names[i] = allocated_name.get();
            impl_->input_names_char[i] = impl_->input_node_names[i].c_str();
        }
        
        // 获取输出节点名称
        // Query output node names
        size_t num_outputs = impl_->session->GetOutputCount();
        impl_->output_node_names.resize(num_outputs);
        impl_->output_names_char.resize(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto allocated_name = impl_->session->GetOutputNameAllocated(i, allocator);
            impl_->output_node_names[i] = allocated_name.get();
            impl_->output_names_char[i] = impl_->output_node_names[i].c_str();
        }
        
        return true;
    } catch (const std::exception& e) {
        ALGO_LOGE(CONFIG, "OnnxBackend Load failed: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOGE(CONFIG, "OnnxBackend Load failed with unknown exception");
        return false;
    }
}

bool OnnxBackend::Run(const std::vector<ModelInput>& inputs,
                      std::vector<ModelOutput>* outputs) {
    if (!impl_->session) {
        ALGO_LOGE(BACKEND, "OnnxBackend Run failed: Session not loaded.");
        return false;
    }
    // 校验输入数量与模型期望一致
    // Verify input count matches model expectations
    if (inputs.size() != impl_->input_node_names.size()) {
        ALGO_LOGE(BACKEND, "OnnxBackend Run failed: Input size mismatch (expected %d, got %d).",
                  static_cast<int>(impl_->input_node_names.size()), static_cast<int>(inputs.size()));
        return false;
    }
    
    try {
        // ---- 构建 ONNX Runtime 输入张量（零拷贝：使用外部缓冲区）----
        // ---- Build input tensors (zero-copy: use external buffers) ----
        std::vector<Ort::Value> ort_inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                impl_->memory_info,
                const_cast<float*>(static_cast<const float*>(inputs[i].data)),  // 零拷贝：不复制数据 / zero-copy: no data copy
                inputs[i].size,
                inputs[i].shape,
                inputs[i].shape_len
            );
            ort_inputs.push_back(std::move(input_tensor));
        }
        
        // 执行推理 / Run inference
        auto ort_outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names_char.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            impl_->output_names_char.data(),
            impl_->output_names_char.size()
        );
        
        // 将 ORT 输出拷贝到 ModelOutput 结构体
        // Copy ORT outputs into ModelOutput structs
        if (outputs) {
            outputs->resize(ort_outputs.size());
            for (size_t i = 0; i < ort_outputs.size(); ++i) {
                auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
                auto shape = type_info.GetShape();
                size_t num_elements = type_info.GetElementCount();
                const float* raw_data = ort_outputs[i].GetTensorData<float>();
                
                (*outputs)[i].name = impl_->output_node_names[i];
                (*outputs)[i].buffer.assign(raw_data, raw_data + num_elements);
                
                // 填充 shape（最多 4 维，不足补 0）
                // Fill shape (up to 4 dims, pad with 0)
                std::memset((*outputs)[i].shape, 0, sizeof((*outputs)[i].shape));
                for (size_t d = 0; d < std::min(shape.size(), static_cast<size_t>(4)); ++d) {
                    (*outputs)[i].shape[d] = shape[d];
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        ALGO_LOGE(BACKEND, "OnnxBackend Run failed: %s", e.what());
        return false;
    } catch (...) {
        ALGO_LOGE(BACKEND, "OnnxBackend Run failed with unknown exception");
        return false;
    }
}

} // namespace face_rec
