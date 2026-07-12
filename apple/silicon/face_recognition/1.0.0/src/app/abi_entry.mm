/**
 * @file abi_entry.mm
 * @brief C ABI 入口层 — 对外暴露的 C 接口函数（Objective-C++ 实现）
 *        C ABI entry layer — externally exposed C interface functions (Objective-C++ implementation)
 *
 * 该文件实现了人脸识别算法库的 C 语言 ABI 接口，供引擎通过 dlopen/dlsym 动态加载调用。
 * 每个函数都包含 NSException + C++ 异常双重保护，确保 ObjC 和纯 C++ 异常都不会泄漏。
 *
 * This file implements the C ABI interface for the face recognition algorithm library,
 * designed for dynamic loading via dlopen/dlsym by the engine.
 * Every function includes NSException + C++ exception double protection,
 * ensuring neither ObjC nor pure C++ exceptions leak through.
 */
#include "pipeline/algorithm_context.h"
#import <Foundation/Foundation.h>
#include "common/logger.h"
#include "algo/abi_contract.h"
#include <exception>
#include <cstring>
#include <cstdlib>

extern "C" {

/**
 * @brief 初始化算法上下文并加载模型
 *        Initialize algorithm context and load models
 * @param config_json JSON 配置字符串（模型路径、阈值等）/ JSON config string (model paths, thresholds, etc.)
 * @return 算法句柄，失败返回 nullptr / algorithm handle, nullptr on failure
 */
algo_handle_t detector_init(const char *config_json) {
    try {
        @try {
            if (!config_json) return nullptr;
            auto *ctx = new face_rec::AlgorithmContext();
            if (!ctx->Initialize(config_json)) {
                delete ctx;
                return nullptr;
            }
            return reinterpret_cast<algo_handle_t>(ctx);
        } @catch (NSException *e) {
            ALGO_LOGE(APP, "detector_init NSException: %s", [[e reason] UTF8String]);
            return nullptr;
        }
    } catch (const std::exception &e) {
        ALGO_LOGE(APP, "detector_init C++ exception: %s", e.what());
        return nullptr;
    } catch (...) {
        ALGO_LOGE(APP, "detector_init unknown C++ exception");
        return nullptr;
    }
}

/**
 * @brief 对输入图像执行人脸检测与识别推理
 *        Run face detection and recognition inference on the input image
 * @param handle 算法句柄 / algorithm handle
 * @param input 硬件缓冲区描述（输入图像）/ hardware buffer descriptor (input image)
 * @param context_json JSON 上下文（如需要识别的场景 ID）/ JSON context (e.g. scene ID for recognition)
 * @param result 输出推理结果结构体 / output inference result struct
 * @return 0 表示成功，负数表示错误码 / 0 on success, negative on error
 */
int detector_infer(algo_handle_t handle, const hw_buffer_desc_t *input,
                   const char *context_json, infer_result_t *result) {
    try {
        @try {
            if (!handle || !input || !result) return -1;
            auto *ctx = reinterpret_cast<face_rec::AlgorithmContext *>(handle);
            return ctx->Infer(input, context_json, result);
        } @catch (NSException *e) {
            ALGO_LOGE(APP, "detector_infer NSException: %s", [[e reason] UTF8String]);
            return -5;
        }
    } catch (const std::exception &e) {
        ALGO_LOGE(APP, "detector_infer C++ exception: %s", e.what());
        return -5;
    } catch (...) {
        ALGO_LOGE(APP, "detector_infer unknown C++ exception");
        return -5;
    }
}

/**
 * @brief 销毁算法上下文，释放所有资源
 *        Destroy the algorithm context and release all resources
 * @param handle 算法句柄 / algorithm handle
 */
void detector_destroy(algo_handle_t handle) {
    try {
        @try {
            if (handle) {
                auto *ctx = reinterpret_cast<face_rec::AlgorithmContext *>(handle);
                delete ctx;
            }
        } @catch (NSException *e) {
            ALGO_LOGE(APP, "detector_destroy NSException: %s", [[e reason] UTF8String]);
        }
    } catch (const std::exception &e) {
        ALGO_LOGE(APP, "detector_destroy C++ exception: %s", e.what());
    } catch (...) {
        ALGO_LOGE(APP, "detector_destroy unknown C++ exception");
    }
}

/** @brief 返回算法库版本号 / Return algorithm library version string */
const char *detector_version(void) {
    return "1.0.0";
}

/** @brief 返回算法库名称 / Return algorithm library name string */
const char *detector_name(void) {
    return "face_recognition_m1_pro";
}

/**
 * @brief 自检函数：验证 ABI 结构体尺寸和必要符号
 *        Self-test: verify ABI struct sizes and required symbols
 * @return 0 表示通过，负数表示失败 / 0 on pass, negative on failure
 */
int detector_self_test(void) {
    try {
        @try {
            // 1. Verify C ABI struct sizes match engine expectations
            // 验证 C ABI 结构体尺寸与引擎期望一致
            if (sizeof(hw_buffer_desc_t) != 144) {
                ALGO_LOGE(APP, "self_test: hw_buffer_desc_t size %zu != 144", sizeof(hw_buffer_desc_t));
                return -1;
            }
            if (sizeof(infer_result_t) != 40) {
                ALGO_LOGE(APP, "self_test: infer_result_t size %zu != 40", sizeof(infer_result_t));
                return -2;
            }

            // 2. Verify version/name strings
            // 验证版本号与名称字符串
            const char* ver = detector_version();
            const char* name = detector_name();
            if (!ver || !name) {
                ALGO_LOGE(APP, "self_test: version or name string is null");
                return -3;
            }

            ALGO_LOGI(APP, "self_test passed: name=%s version=%s", name, ver);
            return 0;
        } @catch (NSException *e) {
            ALGO_LOGE(APP, "self_test NSException: %s", [[e reason] UTF8String]);
            return -5;
        }
    } catch (const std::exception &e) {
        ALGO_LOGE(APP, "self_test C++ exception: %s", e.what());
        return -5;
    } catch (...) {
        ALGO_LOGE(APP, "self_test unknown exception");
        return -5;
    }
}

/**
 * @brief 更新人脸库（添加/删除/修改注册人脸）
 *        Update the face library (add/delete/modify registered faces)
 * @param handle 算法句柄 / algorithm handle
 * @param face_library_json JSON 格式的人脸库更新数据 / face library update data in JSON format
 * @return 0 表示成功，负数表示错误码 / 0 on success, negative on error
 */
int detector_update_face_library(algo_handle_t handle, const char *face_library_json) {
    try {
        @try {
            if (!handle || !face_library_json) return -1;
            auto *ctx = reinterpret_cast<face_rec::AlgorithmContext *>(handle);
            return ctx->UpdateFaceLibrary(face_library_json);
        } @catch (NSException *e) {
            ALGO_LOGE(APP, "detector_update_face_library NSException: %s", [[e reason] UTF8String]);
            return -5;
        }
    } catch (const std::exception &e) {
        ALGO_LOGE(APP, "detector_update_face_library C++ exception: %s", e.what());
        return -5;
    } catch (...) {
        ALGO_LOGE(APP, "detector_update_face_library unknown C++ exception");
        return -5;
    }
}

/**
 * @brief 释放推理结果中动态分配的内存
 *        Free dynamically allocated memory in the inference result
 * @param result 推理结果结构体指针 / pointer to inference result struct
 */
void algo_free_result(infer_result_t *result) {
    if (result && result->result_json) {
        std::free(result->result_json);
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}

/** @brief algo_free_result 的别名（保持 ABI 兼容性）/ Alias for algo_free_result (ABI compatibility) */
void detector_free_result(infer_result_t *result) {
    algo_free_result(result);
}

} // extern "C"
