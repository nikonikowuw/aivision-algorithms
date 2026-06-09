#ifndef ORT_COREML_H
#define ORT_COREML_H

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>
#include <string>
#include <unordered_map>

namespace face_recognition {

// CoreML 配置模式
enum class CoreMLFormat {
    MLProgram,      // MLProgram 格式，适用于大多数模型
    NeuralNetwork,  // NeuralNetwork 格式，适用于 2d106det 等不兼容 MLProgram 的模型
};

// 创建强制 CoreML EP 的 SessionOptions
// 禁用 CPU EP fallback，如果模型有任何节点不能被 CoreML 接管则初始化失败
inline Ort::SessionOptions CreateCoreMLSessionOptions(CoreMLFormat format) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    opts.AddFreeDimensionOverrideByName("batch", 1);
    opts.AddFreeDimensionOverrideByName("batch_size", 1);

    std::string model_format = (format == CoreMLFormat::MLProgram) ? "MLProgram" : "NeuralNetwork";
    std::unordered_map<std::string, std::string> coreml_options = {
        {"ModelFormat", model_format},
        {"MLComputeUnits", "ALL"},
        {"RequireStaticInputShapes", "0"},
        {"EnableOnSubgraphs", "1"},
    };
    opts.AppendExecutionProvider("CoreML", coreml_options);
    return opts;
}

// 直接配置 SessionOptions 的 CoreML EP
inline void ConfigureCoreMLSessionOptions(Ort::SessionOptions& opts, CoreMLFormat format) {
    std::string model_format = (format == CoreMLFormat::MLProgram) ? "MLProgram" : "NeuralNetwork";
    std::unordered_map<std::string, std::string> coreml_options = {
        {"ModelFormat", model_format},
        {"MLComputeUnits", "ALL"},
        {"RequireStaticInputShapes", "0"},
        {"EnableOnSubgraphs", "1"},
    };
    opts.AppendExecutionProvider("CoreML", coreml_options);
}

} // namespace face_recognition

#endif // ORT_COREML_H
