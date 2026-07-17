// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Native CoreML backend implementation.

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include "coreml_backend.h"

#include <cstring>
#include <sstream>

namespace smoking {

struct CoreMLNativeBackend::Impl {
    MLModel* model = nil;
    NSString* input_name = nil;
    NSString* output_name = nil;
    int32_t input_width = 0;
    int32_t input_height = 0;
};

CoreMLNativeBackend::CoreMLNativeBackend() : impl_(std::make_unique<Impl>()) {}
CoreMLNativeBackend::~CoreMLNativeBackend() = default;

bool CoreMLNativeBackend::Load(const ModelSpec& spec, std::string& error) {
    error.clear();
    @autoreleasepool {
        @try {
            if (spec.model_path.empty()) {
                error = "CoreML model path is empty";
                return false;
            }
            NSString* path = [NSString stringWithUTF8String:spec.model_path.c_str()];
            BOOL is_directory = NO;
            if (!path || ![[NSFileManager defaultManager] fileExistsAtPath:path
                                                               isDirectory:&is_directory] ||
                !is_directory || ![[path pathExtension] isEqualToString:@"mlmodelc"]) {
                error = "compiled CoreML model bundle is missing or invalid: " +
                        spec.model_path;
                return false;
            }

            MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
            configuration.computeUnits = MLComputeUnitsAll;
            NSError* ns_error = nil;
            MLModel* model = [MLModel modelWithContentsOfURL:
                                  [NSURL fileURLWithPath:path]
                                                configuration:configuration
                                                        error:&ns_error];
            if (!model || ns_error) {
                error = ns_error ? [[ns_error localizedDescription] UTF8String]
                                 : "CoreML returned a null model";
                return false;
            }

            MLModelDescription* description = model.modelDescription;
            NSDictionary<NSString*, MLFeatureDescription*>* inputs =
                description.inputDescriptionsByName;
            NSDictionary<NSString*, MLFeatureDescription*>* outputs =
                description.outputDescriptionsByName;
            if (inputs.count != 1 || outputs.count != 1) {
                error = "CoreML DAMO model must expose exactly one input and one output";
                return false;
            }

            NSString* input_name = inputs.allKeys.firstObject;
            NSString* output_name = outputs.allKeys.firstObject;
            MLFeatureDescription* input = inputs[input_name];
            MLFeatureDescription* output = outputs[output_name];
            if (input.type != MLFeatureTypeImage || !input.imageConstraint) {
                error = "CoreML model input must be an image feature";
                return false;
            }
            if (input.imageConstraint.pixelsWide != spec.input_width ||
                input.imageConstraint.pixelsHigh != spec.input_height) {
                std::ostringstream stream;
                stream << "CoreML image input is " << input.imageConstraint.pixelsWide
                       << "x" << input.imageConstraint.pixelsHigh << ", expected "
                       << spec.input_width << "x" << spec.input_height;
                error = stream.str();
                return false;
            }
            if (output.type != MLFeatureTypeMultiArray || !output.multiArrayConstraint ||
                output.multiArrayConstraint.dataType != MLMultiArrayDataTypeFloat32) {
                error = "CoreML DAMO output must be a Float32 multi-array";
                return false;
            }

            impl_->model = model;
            impl_->input_name = input_name;
            impl_->output_name = output_name;
            impl_->input_width = spec.input_width;
            impl_->input_height = spec.input_height;
            return true;
        } @catch (NSException* exception) {
            error = std::string("CoreML load NSException: ") +
                    [[exception description] UTF8String];
            return false;
        }
    }
}

bool CoreMLNativeBackend::Run(uint64_t pixel_buffer_handle,
                              InferenceTensor& output,
                              std::string& error) {
    output.values.clear();
    output.shape.clear();
    error.clear();
    @autoreleasepool {
        @try {
            if (!impl_->model || pixel_buffer_handle == 0) {
                error = "CoreML backend is not loaded or input is null";
                return false;
            }
            CVPixelBufferRef pixel_buffer =
                reinterpret_cast<CVPixelBufferRef>(
                    static_cast<uintptr_t>(pixel_buffer_handle));
            if (CVPixelBufferGetWidth(pixel_buffer) !=
                    static_cast<size_t>(impl_->input_width) ||
                CVPixelBufferGetHeight(pixel_buffer) !=
                    static_cast<size_t>(impl_->input_height)) {
                error = "prepared CoreML input dimensions do not match the model";
                return false;
            }

            MLFeatureValue* image_value =
                [MLFeatureValue featureValueWithPixelBuffer:pixel_buffer];
            NSError* ns_error = nil;
            MLDictionaryFeatureProvider* provider =
                [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{impl_->input_name : image_value}
                                 error:&ns_error];
            if (!provider || ns_error) {
                error = ns_error ? [[ns_error localizedDescription] UTF8String]
                                 : "failed to create CoreML feature provider";
                return false;
            }

            id<MLFeatureProvider> prediction =
                [impl_->model predictionFromFeatures:provider error:&ns_error];
            if (!prediction || ns_error) {
                error = ns_error ? [[ns_error localizedDescription] UTF8String]
                                 : "CoreML prediction returned null";
                return false;
            }

            MLMultiArray* array =
                [[prediction featureValueForName:impl_->output_name] multiArrayValue];
            if (!array || array.dataType != MLMultiArrayDataTypeFloat32) {
                error = "CoreML prediction output is not a Float32 multi-array";
                return false;
            }

            for (NSNumber* dimension in array.shape) {
                output.shape.push_back(dimension.longLongValue);
            }
            if ((output.shape.size() != 2 && output.shape.size() != 3) ||
                output.shape.back() != 6 ||
                (output.shape.size() == 3 && output.shape.front() != 1)) {
                error = "CoreML DAMO output shape must be [N,6] or [1,N,6]";
                output.shape.clear();
                return false;
            }

            const NSArray<NSNumber*>* strides = array.strides;
            int64_t expected_stride = 1;
            for (NSInteger i = static_cast<NSInteger>(output.shape.size()) - 1;
                 i >= 0; --i) {
                if (strides[static_cast<NSUInteger>(i)].longLongValue !=
                    expected_stride) {
                    error = "CoreML DAMO output must be contiguous";
                    output.shape.clear();
                    return false;
                }
                expected_stride *= output.shape[static_cast<size_t>(i)];
            }

            output.values.resize(array.count);
            std::memcpy(output.values.data(), array.dataPointer,
                        output.values.size() * sizeof(float));
            return true;
        } @catch (NSException* exception) {
            output.values.clear();
            output.shape.clear();
            error = std::string("CoreML prediction NSException: ") +
                    [[exception description] UTF8String];
            return false;
        }
    }
}

std::string CoreMLNativeBackend::BackendInfo() const {
    return "coreml_native (MLComputeUnitsAll, image input)";
}

}  // namespace smoking
