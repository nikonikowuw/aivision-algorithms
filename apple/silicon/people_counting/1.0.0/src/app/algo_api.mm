/**
 * @file algo_api.mm
 * @brief C ABI entry layer for people_counting algorithm library
 */

#include "pipeline/pipeline.h"
#import <Foundation/Foundation.h>
#include "common/logger.h"
#include "engine/include/algo/abi_contract.h"
#include <exception>
#include <cstring>
#include <cstdlib>

extern "C" {

/**
 * @brief Initialize algorithm context and load model
 * @param config_json JSON config string
 * @return algorithm handle, nullptr on failure
 */
algo_handle_t detector_init(const char *config_json) {
    try {
        @try {
            if (!config_json) return nullptr;
            auto *ctx = new people_count::PipelineOrchestrator();
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
 * @brief Run inference on the input image
 * @param handle algorithm handle
 * @param input hardware buffer descriptor (input image)
 * @param context_json JSON context (e.g. crossing lines configuration)
 * @param result output inference result struct
 * @return 0 on success, negative on error
 */
int detector_infer(algo_handle_t handle, const hw_buffer_desc_t *input,
                   const char *context_json, infer_result_t *result) {
    try {
        @try {
            if (!handle || !input || !result) return -1;
            auto *ctx = reinterpret_cast<people_count::PipelineOrchestrator *>(handle);
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
 * @brief Destroy the algorithm context and release all resources
 * @param handle algorithm handle
 */
void detector_destroy(algo_handle_t handle) {
    try {
        @try {
            if (handle) {
                auto *ctx = reinterpret_cast<people_count::PipelineOrchestrator *>(handle);
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

/** @brief Return algorithm library version string */
const char *detector_version(void) {
    return "1.0.0";
}

/** @brief Return algorithm library name string */
const char *detector_name(void) {
    return "people_counting";
}

/**
 * @brief Self-test: verify ABI struct sizes and required symbols
 * @return 0 on pass, negative on failure
 */
int detector_self_test(void) {
    try {
        @try {
            if (sizeof(hw_buffer_desc_t) != 144) {
                ALGO_LOGE(APP, "self_test: hw_buffer_desc_t size %zu != 144", sizeof(hw_buffer_desc_t));
                return -1;
            }
            if (sizeof(infer_result_t) != 40) {
                ALGO_LOGE(APP, "self_test: infer_result_t size %zu != 40", sizeof(infer_result_t));
                return -2;
            }

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
 * @brief Free dynamically allocated memory in the inference result
 * @param result pointer to inference result struct
 */
void algo_free_result(infer_result_t *result) {
    if (result && result->result_json) {
        std::free(result->result_json);
        result->result_json = nullptr;
        result->result_json_len = 0;
    }
}

void detector_free_result(infer_result_t *result) {
    algo_free_result(result);
}

} // extern "C"
