/**
 * @file coreml_backend.mm
 * @brief Apple CoreML 推理后端实现（Objective-C++）
 *        Apple CoreML inference backend implementation (Objective-C++)
 *
 * 该文件实现了 CoreML 模型的编译加载与推理：
 * - 通过 MLModel.compileModelAtURL 编译 .mlmodel → .mlmodelc
 * - 使用 MLMultiArray 的 dataPointer 实现输入数据的零拷贝包装
 * - 调用 MLModel predictionFromFeatures 执行推理
 *
 * This file implements CoreML model compilation, loading, and inference:
 * - Compile .mlmodel → .mlmodelc via MLModel.compileModelAtURL
 * - Zero-copy input wrapping via MLMultiArray dataPointer
 * - Run inference via MLModel predictionFromFeatures
 */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "inference_backend.h"
#include "common/logger.h"
#include <cstring>
#include <algorithm>

namespace face_rec {

/**
 * @struct CoreMLBackend::Impl
 * @brief CoreML 后端的内部实现（PIMPL 模式）
 *        Internal implementation of CoreMLBackend (PIMPL pattern)
 *
 * 持有 MLModel 实例及输入输出节点名称 NSString，
 * 避免在头文件中暴露 Objective-C 类型。
 * Holds the MLModel instance and input/output node names as NSString,
 * avoiding exposure of Objective-C types in the header.
 */
struct CoreMLBackend::Impl {
    MLModel *model = nil;         ///< CoreML 模型实例 / CoreML model instance
    NSString *input_name = nil;   ///< 输入层名称 / input layer name
    NSString *output_name = nil;  ///< 输出层名称 / output layer name
};

CoreMLBackend::CoreMLBackend() : impl_(std::make_unique<Impl>()) {}
CoreMLBackend::~CoreMLBackend() {
    Unload();
}

void CoreMLBackend::Unload() {
    if (impl_->model) {
        impl_->model = nil;
    }
}

bool CoreMLBackend::Load(const std::string& model_path) {
    @try {
        NSString *ns_path = [NSString stringWithUTF8String:model_path.c_str()];
        NSURL *model_url = [NSURL fileURLWithPath:ns_path];
        
        NSError *error = nil;
        NSURL *compiled_url = model_url;
        if (![[ns_path pathExtension] isEqualToString:@"mlmodelc"]) {
            compiled_url = [MLModel compileModelAtURL:model_url error:&error];
            if (error || !compiled_url) {
                ALGO_LOGE(CONFIG, "CoreML compilation failed: %s",
                          error ? [[error localizedDescription] UTF8String] : "Unknown error");
                return false;
            }
        }
        
        // Step 2: 创建 MLModelConfiguration 并加载已编译的模型
        // Step 2: Create MLModelConfiguration and load the compiled model
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;  ///< 使用所有计算单元（CPU+GPU+ANE）
        
        impl_->model = [MLModel modelWithContentsOfURL:compiled_url configuration:config error:&error];
        if (error || !impl_->model) {
            ALGO_LOGE(CONFIG, "CoreML Load failed: %s", 
                      error ? [[error localizedDescription] UTF8String] : "Unknown error");
            return false;
        }
        
        // Step 3: 读取模型元数据中的输入输出节点名称
        // Step 3: Read input/output node names from model metadata
        MLModelDescription *desc = impl_->model.modelDescription;
        impl_->input_name = [[[desc inputDescriptionsByName] allKeys] firstObject];
        impl_->output_name = [[[desc outputDescriptionsByName] allKeys] firstObject];
        
        ALGO_LOGI(CONFIG, "Successfully loaded CoreML model dynamically: %s", model_path.c_str());
        return true;
    } @catch (NSException *exception) {
        ALGO_LOGE(CONFIG, "CoreML Load exception: %s", [[exception reason] UTF8String]);
        return false;
    }
}

bool CoreMLBackend::Run(const std::vector<ModelInput>& inputs,
                        std::vector<ModelOutput>* outputs) {
    if (!impl_->model) {
        ALGO_LOGE(BACKEND, "CoreML Run failed: Model not loaded.");
        return false;
    }

    if (inputs.empty() || !inputs[0].data || inputs[0].shape_len == 0) {
        ALGO_LOGE(BACKEND, "CoreML Run failed: Empty or invalid inputs.");
        return false;
    }

    @try {
        NSError *error = nil;
        
        // ---- 构建 MLMultiArray 包装输入数据（零拷贝）----
        // ---- Build MLMultiArray wrapping input data (zero-copy) ----
        NSMutableArray<NSNumber *> *shape_ns = [NSMutableArray array];
        NSMutableArray<NSNumber *> *strides_ns = [NSMutableArray array];
        
        // 将 int64_t shape 转为 NSNumber 数组
        // Convert int64_t shape to NSNumber array
        int64_t total_elements __attribute__((unused)) = 1;
        for (size_t i = 0; i < inputs[0].shape_len; ++i) {
            [shape_ns addObject:@(inputs[0].shape[i])];
            total_elements *= inputs[0].shape[i];
        }
        
        // 计算各维度的 stride（用于构建 MLMultiArray 视图）
        // Calculate strides for each dimension (to build the MLMultiArray view)
        int64_t stride = 1;
        for (int i = static_cast<int>(inputs[0].shape_len) - 1; i >= 0; --i) {
            [strides_ns insertObject:@(stride) atIndex:0];
            stride *= inputs[0].shape[i];
        }
        
        // initWithDataPointer 实现零拷贝：直接使用预分配的 float 缓冲区
        // initWithDataPointer achieves zero-copy: use the pre-allocated float buffer directly
        MLMultiArray *input_array = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(inputs[0].data)
                                                                       shape:shape_ns
                                                                    dataType:MLMultiArrayDataTypeFloat32
                                                                     strides:strides_ns
                                                                 deallocator:^(void *bytes) { (void)bytes; }  ///< 不释放外部内存（no-op deallocator）
                                                                       error:&error];
        if (error || !input_array) {
            ALGO_LOGE(BACKEND, "CoreML input wrapper creation failed: %s", 
                      error ? [[error localizedDescription] UTF8String] : "Unknown error");
            return false;
        }
        
        // 包装为 MLDictionaryFeatureProvider 作为模型输入
        // Wrap as MLDictionaryFeatureProvider for model input
        MLDictionaryFeatureProvider *feature_provider = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{impl_->input_name : input_array} error:&error];
            
        // 执行预测 / Run prediction
        id<MLFeatureProvider> results = [impl_->model predictionFromFeatures:feature_provider error:&error];
        if (error || !results) {
            ALGO_LOGE(BACKEND, "CoreML Prediction failed: %s", 
                      error ? [[error localizedDescription] UTF8String] : "Unknown error");
            return false;
        }
        
        // 将 CoreML 输出转换为 ModelOutput 结构体
        // Convert CoreML outputs to ModelOutput structs
        if (outputs) {
            NSArray<NSString *> *feature_names = [[results featureNames] allObjects];
            outputs->resize([feature_names count]);
            for (NSUInteger i = 0; i < [feature_names count]; ++i) {
                NSString *out_name = [feature_names objectAtIndex:i];
                MLFeatureValue *val = [results featureValueForName:out_name];
                MLMultiArray *out_arr = [val multiArrayValue];
                if (!out_arr || [out_arr dataType] != MLMultiArrayDataTypeFloat32) {
                    ALGO_LOGE(BACKEND, "CoreML output is not a Float32 MLMultiArray: %s",
                              [out_name UTF8String]);
                    return false;
                }

                ModelOutput& out_obj = (*outputs)[i];
                out_obj.name = [out_name UTF8String];
                int64_t count = [out_arr count];
                out_obj.buffer.resize(count);
                
                // 从 MLMultiArray dataPointer 拷贝输出数据
                // Copy output data from MLMultiArray dataPointer
                float *out_ptr = (float *)[out_arr dataPointer];
                std::memcpy(out_obj.buffer.data(), out_ptr, count * sizeof(float));
                
                // 填充输出 shape（最多 4 维）
                // Fill output shape (up to 4 dims)
                std::memset(out_obj.shape, 0, sizeof(out_obj.shape));
                NSArray *out_shape = [out_arr shape];
                for (size_t d = 0; d < std::min([out_shape count], static_cast<NSUInteger>(4)); ++d) {
                    out_obj.shape[d] = [[out_shape objectAtIndex:d] longLongValue];
                }
                
            }
        }
        
        return true;
    } @catch (NSException *exception) {
        ALGO_LOGE(BACKEND, "CoreML prediction exception: %s", [[exception reason] UTF8String]);
        return false;
    }
}

} // namespace face_rec
