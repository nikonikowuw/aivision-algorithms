/**
 * @file coreml_backend.mm
 * @brief CoreML inference backend implementation (Objective-C++)
 */

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_backend.h"
#include "common/logger.h"
#include <cstring>
#include <algorithm>

namespace people_count {

struct CoreMLBackend::Impl {
    MLModel *model = nil;
    NSString *input_name = nil;
    NSString *output_name = nil;
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
                ALGO_LOGE(BACKEND, "CoreML compilation failed: %s",
                          error ? [[error localizedDescription] UTF8String] : "Unknown error");
                return false;
            }
        }
        
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;
        
        impl_->model = [MLModel modelWithContentsOfURL:compiled_url configuration:config error:&error];
        if (error || !impl_->model) {
            ALGO_LOGE(BACKEND, "CoreML Load failed: %s", 
                      error ? [[error localizedDescription] UTF8String] : "Unknown error");
            return false;
        }
        
        MLModelDescription *desc = impl_->model.modelDescription;
        impl_->input_name = [[[desc inputDescriptionsByName] allKeys] firstObject];
        impl_->output_name = [[[desc outputDescriptionsByName] allKeys] firstObject];
        
        ALGO_LOGI(BACKEND, "Successfully loaded CoreML model: %s", model_path.c_str());
        return true;
    } @catch (NSException *exception) {
        ALGO_LOGE(BACKEND, "CoreML Load exception: %s", [[exception reason] UTF8String]);
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
        
        NSMutableArray<NSNumber *> *shape_ns = [NSMutableArray array];
        NSMutableArray<NSNumber *> *strides_ns = [NSMutableArray array];
        
        int64_t total_elements = 1;
        for (size_t i = 0; i < inputs[0].shape_len; ++i) {
            [shape_ns addObject:@(inputs[0].shape[i])];
            total_elements *= inputs[0].shape[i];
        }
        
        int64_t stride = 1;
        for (int i = static_cast<int>(inputs[0].shape_len) - 1; i >= 0; --i) {
            [strides_ns insertObject:@(stride) atIndex:0];
            stride *= inputs[0].shape[i];
        }
        
        MLMultiArray *input_array = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(inputs[0].data)
                                                                       shape:shape_ns
                                                                    dataType:MLMultiArrayDataTypeFloat32
                                                                     strides:strides_ns
                                                                 deallocator:^(void *bytes) { (void)bytes; }
                                                                       error:&error];
        if (error || !input_array) {
            ALGO_LOGE(BACKEND, "CoreML input wrapper creation failed: %s", 
                      error ? [[error localizedDescription] UTF8String] : "Unknown error");
            return false;
        }
        
        MLDictionaryFeatureProvider *feature_provider = [[MLDictionaryFeatureProvider alloc] 
            initWithDictionary:@{impl_->input_name : input_array} error:&error];
            
        id<MLFeatureProvider> results = [impl_->model predictionFromFeatures:feature_provider error:&error];
        if (error || !results) {
            ALGO_LOGE(BACKEND, "CoreML Prediction failed: %s", 
                      error ? [[error localizedDescription] UTF8String] : "Unknown error");
            return false;
        }
        
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
                
                float *out_ptr = (float *)[out_arr dataPointer];
                std::memcpy(out_obj.buffer.data(), out_ptr, count * sizeof(float));
                
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

} // namespace people_count
