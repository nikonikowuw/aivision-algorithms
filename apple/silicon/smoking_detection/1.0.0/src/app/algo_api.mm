// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Defensive C ABI entry points.

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <algo/abi_contract.h>
#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common/config.h"
#include "../common/event_json.h"
#include "../common/logger.h"
#include "../pipeline/pipeline.h"
#include "../runtime/apple_buffer_view.h"
#include "picojson.h"

namespace smoking {
namespace {

struct HandleState {
    std::mutex mutex;
    std::unique_ptr<SmokingPipeline> pipeline;
};

std::mutex g_registry_mutex;
std::unordered_map<uintptr_t, std::shared_ptr<HandleState>> g_registry;
std::atomic<uintptr_t> g_next_handle{1};

void ClearResult(infer_result_t* result) {
    if (!result) return;
    result->result_json = nullptr;
    result->result_json_len = 0;
    result->infer_time_us = 0;
    std::memset(result->reserved, 0, sizeof(result->reserved));
}

uint32_t ElapsedMicroseconds(
    const std::chrono::steady_clock::time_point& started) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
    return static_cast<uint32_t>(std::min<int64_t>(
        std::max<int64_t>(elapsed, 0), std::numeric_limits<uint32_t>::max()));
}

std::string ModuleDirectory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&detector_init), &info) == 0 ||
        !info.dli_fname) {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
}

std::shared_ptr<HandleState> LookupHandle(algo_handle_t handle) {
    const uintptr_t token = reinterpret_cast<uintptr_t>(handle);
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    const auto it = g_registry.find(token);
    return it == g_registry.end() ? nullptr : it->second;
}

algo_handle_t RegisterHandle(std::shared_ptr<HandleState> state) {
    uintptr_t token = g_next_handle.fetch_add(1, std::memory_order_relaxed);
    if (token == 0) token = g_next_handle.fetch_add(1, std::memory_order_relaxed);
    token = (token << 1u) | 1u;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_registry.emplace(token, std::move(state));
    return reinterpret_cast<algo_handle_t>(token);
}

std::shared_ptr<HandleState> RemoveHandle(algo_handle_t handle) {
    const uintptr_t token = reinterpret_cast<uintptr_t>(handle);
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    const auto it = g_registry.find(token);
    if (it == g_registry.end()) return nullptr;
    auto state = it->second;
    g_registry.erase(it);
    return state;
}

ErrorCode ValidateBgr24(const hw_buffer_desc_t& input,
                        std::string& error) {
    if (input.buffer_type != HW_BUFFER_TYPE_DEFAULT ||
        input.buffer_owner != HW_BUFFER_OWNER_ENGINE || !input.data ||
        input.width == 0 || input.height == 0 ||
        input.width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max() / 3) ||
        input.height > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        input.stride > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        input.pixel_format != kPixelFormatBGR24) {
        error = "invalid Engine-owned packed BGR24 descriptor";
        return ErrorCode::kInvalidBuffer;
    }
    const size_t row_bytes = static_cast<size_t>(input.width) * 3u;
    if (input.stride < row_bytes) {
        error = "BGR24 stride is smaller than width * 3";
        return ErrorCode::kInvalidBuffer;
    }
    const size_t rows_before_last = static_cast<size_t>(input.height - 1u);
    if (rows_before_last > (std::numeric_limits<size_t>::max() - row_bytes) /
                               input.stride) {
        error = "BGR24 buffer size calculation overflow";
        return ErrorCode::kInvalidBuffer;
    }
    const size_t required = rows_before_last * input.stride + row_bytes;
    if (input.size < required) {
        error = "BGR24 buffer size is smaller than the described frame";
        return ErrorCode::kInvalidBuffer;
    }
    return ErrorCode::kSuccess;
}

bool AllocateResult(const std::string& json, infer_result_t& result) {
    if (json.size() == std::numeric_limits<size_t>::max()) return false;
    char* memory = static_cast<char*>(std::malloc(json.size() + 1u));
    if (!memory) return false;
    std::memcpy(memory, json.data(), json.size());
    memory[json.size()] = '\0';
    result.result_json = memory;
    result.result_json_len = json.size();
    return true;
}

bool LoadTestImageBgr(const std::filesystem::path& path,
                      std::vector<uint8_t>& bgr, int32_t& width,
                      int32_t& height, std::string& error) {
    bgr.clear();
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:path.string().c_str()]];
        CGImageSourceRef source = CGImageSourceCreateWithURL(
            (__bridge CFURLRef)url, nullptr);
        if (!source) {
            error = "failed to open testimage.jpg";
            return false;
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
        CFRelease(source);
        if (!image) {
            error = "failed to decode testimage.jpg";
            return false;
        }
        const size_t image_width = CGImageGetWidth(image);
        const size_t image_height = CGImageGetHeight(image);
        if (image_width == 0 || image_height == 0 ||
            image_width > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            image_height > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            image_width > std::numeric_limits<size_t>::max() / image_height / 4u) {
            CGImageRelease(image);
            error = "test image dimensions are invalid";
            return false;
        }

        std::vector<uint8_t> rgba(image_width * image_height * 4u);
        CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            rgba.data(), image_width, image_height, 8, image_width * 4u,
            color_space, kCGImageAlphaPremultipliedLast |
                             kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(color_space);
        if (!context) {
            CGImageRelease(image);
            error = "failed to create test image bitmap context";
            return false;
        }
        CGContextDrawImage(context, CGRectMake(0, 0, image_width, image_height),
                           image);
        CGContextRelease(context);
        CGImageRelease(image);

        bgr.resize(image_width * image_height * 3u);
        for (size_t index = 0; index < image_width * image_height; ++index) {
            bgr[index * 3u] = rgba[index * 4u + 2u];
            bgr[index * 3u + 1u] = rgba[index * 4u + 1u];
            bgr[index * 3u + 2u] = rgba[index * 4u];
        }
        width = static_cast<int32_t>(image_width);
        height = static_cast<int32_t>(image_height);
        return true;
    }
}

int RunSelfTest() {
    const std::filesystem::path runtime_root(ModuleDirectory());
    const auto test_image = runtime_root / "testimage.jpg";
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(test_image, filesystem_error) ||
        filesystem_error || std::filesystem::file_size(test_image, filesystem_error) == 0 ||
        filesystem_error) {
        return static_cast<int>(ErrorCode::kInvalidParam);
    }

    std::vector<uint8_t> bgr;
    int32_t width = 0;
    int32_t height = 0;
    std::string error;
    if (!LoadTestImageBgr(test_image, bgr, width, height, error)) {
        return static_cast<int>(ErrorCode::kInvalidBuffer);
    }

    algo_handle_t handle = detector_init("{}");
    if (!handle) return static_cast<int>(ErrorCode::kInferenceFailed);

    hw_buffer_desc_t input{};
    input.size = bgr.size();
    input.width = static_cast<uint32_t>(width);
    input.height = static_cast<uint32_t>(height);
    input.pixel_format = kPixelFormatBGR24;
    input.data = bgr.data();
    input.stride = static_cast<uint32_t>(width * 3);
    input.buffer_type = HW_BUFFER_TYPE_DEFAULT;
    input.buffer_owner = HW_BUFFER_OWNER_ENGINE;

    infer_result_t result{};
    const int infer_status = detector_infer(handle, &input, nullptr, &result);
    bool json_valid = false;
    if (infer_status == 0 && result.result_json &&
        result.result_json_len == std::strlen(result.result_json)) {
        picojson::value root;
        const std::string parse_error = picojson::parse(root, result.result_json);
        json_valid = parse_error.empty() && root.is<picojson::array>();
    }
    algo_free_result(&result);
    detector_destroy(handle);
    return json_valid ? 0 : static_cast<int>(ErrorCode::kSerializationFailed);
}

}  // namespace
}  // namespace smoking

using namespace smoking;

extern "C" {

algo_handle_t detector_init(const char* config_json) {
    @try {
        try {
            AlgoConfig config;
            std::string error;
            if (ParseConfig(config_json, config, error) != ErrorCode::kSuccess) {
                SMOKE_LOG_WARN(LogModule::kApp, "invalid config: %s", error.c_str());
                return nullptr;
            }
            config.runtime_root = ModuleDirectory();
            if (config.runtime_root.empty()) return nullptr;

            auto state = std::make_shared<HandleState>();
            state->pipeline = std::make_unique<SmokingPipeline>();
            const ErrorCode status = state->pipeline->Initialize(config, error);
            if (status != ErrorCode::kSuccess) {
                SMOKE_LOG_WARN(LogModule::kApp, "initialization failed: %s",
                               error.c_str());
                return nullptr;
            }
            return RegisterHandle(std::move(state));
        } catch (const std::exception& exception) {
            SMOKE_LOG_WARN(LogModule::kApp, "detector_init exception: %s",
                           exception.what());
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    } @catch (NSException* exception) {
        SMOKE_LOG_WARN(LogModule::kApp, "detector_init NSException: %s",
                       [[exception description] UTF8String]);
        return nullptr;
    }
}

int detector_infer(algo_handle_t handle, const hw_buffer_desc_t* input,
                   const char* context_json, infer_result_t* result) {
    (void)context_json;
    const auto started = std::chrono::steady_clock::now();
    @try {
        try {
            ClearResult(result);
            if (!handle || !input || !result) {
                return static_cast<int>(ErrorCode::kInvalidParam);
            }
            const auto state = LookupHandle(handle);
            if (!state) return static_cast<int>(ErrorCode::kInvalidParam);

            std::vector<SmokingEvent> events;
            std::string error;
            int32_t frame_width = 0;
            int32_t frame_height = 0;
            ErrorCode status = ErrorCode::kInvalidBuffer;
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->pipeline) {
                return static_cast<int>(ErrorCode::kInvalidParam);
            }

            if (input->buffer_type == HW_BUFFER_TYPE_APPLE_NATIVE) {
                AppleBufferView view;
                status = ValidateAppleBuffer(*input, view, error);
                if (status == ErrorCode::kSuccess) {
                    frame_width = view.width;
                    frame_height = view.height;
                    status = state->pipeline->InferNative(
                        view.native_handle, view.pixel_format, view.width,
                        view.height, events, error);
                }
            } else {
                status = ValidateBgr24(*input, error);
                if (status == ErrorCode::kSuccess) {
                    frame_width = static_cast<int32_t>(input->width);
                    frame_height = static_cast<int32_t>(input->height);
                    status = state->pipeline->Infer(
                        static_cast<const uint8_t*>(input->data), frame_width,
                        frame_height, input->pixel_format,
                        static_cast<int32_t>(input->stride), events, error);
                }
            }
            if (status != ErrorCode::kSuccess) {
                result->infer_time_us = ElapsedMicroseconds(started);
                SMOKE_LOG_WARN(LogModule::kApp, "inference failed: %s",
                               error.c_str());
                return static_cast<int>(status);
            }

            std::string json;
            status = SerializeEvents(events, frame_width, frame_height, json,
                                     error);
            if (status != ErrorCode::kSuccess || !AllocateResult(json, *result)) {
                ClearResult(result);
                result->infer_time_us = ElapsedMicroseconds(started);
                return static_cast<int>(ErrorCode::kSerializationFailed);
            }
            result->infer_time_us = ElapsedMicroseconds(started);
            return 0;
        } catch (const std::exception& exception) {
            ClearResult(result);
            if (result) result->infer_time_us = ElapsedMicroseconds(started);
            SMOKE_LOG_WARN(LogModule::kApp, "detector_infer exception: %s",
                           exception.what());
            return static_cast<int>(ErrorCode::kInternalError);
        } catch (...) {
            ClearResult(result);
            if (result) result->infer_time_us = ElapsedMicroseconds(started);
            return static_cast<int>(ErrorCode::kInternalError);
        }
    } @catch (NSException* exception) {
        ClearResult(result);
        if (result) result->infer_time_us = ElapsedMicroseconds(started);
        SMOKE_LOG_WARN(LogModule::kApp, "detector_infer NSException: %s",
                       [[exception description] UTF8String]);
        return static_cast<int>(ErrorCode::kInternalError);
    }
}

void detector_destroy(algo_handle_t handle) {
    @try {
        try {
            if (!handle) return;
            const auto state = RemoveHandle(handle);
            if (!state) return;
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->pipeline) state->pipeline->Destroy();
            state->pipeline.reset();
        } catch (...) {
        }
    } @catch (NSException*) {
    }
}

const char* detector_version(void) {
    @try {
        try {
            return SmokingPipeline::Version();
        } catch (...) {
            return "";
        }
    } @catch (NSException*) {
        return "";
    }
}

const char* detector_name(void) {
    @try {
        try {
            return SmokingPipeline::Name();
        } catch (...) {
            return "";
        }
    } @catch (NSException*) {
        return "";
    }
}

int detector_self_test(void) {
    @try {
        try {
            return RunSelfTest();
        } catch (...) {
            return static_cast<int>(ErrorCode::kInternalError);
        }
    } @catch (NSException*) {
        return static_cast<int>(ErrorCode::kInternalError);
    }
}

void algo_free_result(infer_result_t* result) {
    @try {
        try {
            if (!result) return;
            std::free(result->result_json);
            ClearResult(result);
        } catch (...) {
        }
    } @catch (NSException*) {
    }
}

}  // extern "C"
