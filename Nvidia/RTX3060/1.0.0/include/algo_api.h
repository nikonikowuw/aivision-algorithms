/**
 * @file algo_api.h
 * @brief GPU Face Recognition Algorithm Package — Public C API Header
 *        (Optional) Exposes the C ABI interface for external consumers.
 * @note  This header mirrors algo_contract.h but is intended as the
 *        public-facing API for the GPU face recognition package.
 */

#ifndef GPU_FACE_RECOGNITION_ALGO_API_H
#define GPU_FACE_RECOGNITION_ALGO_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Opaque algorithm context handle
     */
    typedef struct algo_context_t *algo_handle_t;

    /**
     * @brief Hardware buffer descriptor (matches engine ABI)
     */
    typedef struct
    {
        int dma_fd;
        size_t size;
        uint32_t width;
        uint32_t height;
        uint32_t pixel_format;
        int dma_buf_fd;
        uint64_t phys_addr;
        void *data;
        uint32_t stride;

        uint32_t buffer_type;
        uint32_t buffer_owner;
        uint32_t reserved_flags;

        union {
            uint8_t plat[80];
            uint64_t plat_u64[10];
        };

    } hw_buffer_desc_t; // sizeof = 144

    /**
     * @brief Inference result structure
     */
    typedef struct
    {
        char *result_json;
        size_t result_json_len;
        uint32_t infer_time_us;
        int reserved[4];
    } infer_result_t; // sizeof = 40

    /** @brief Initialize algorithm context */
    algo_handle_t detector_init(const char *config_json);

    /** @brief Run inference on one frame */
    int detector_infer(algo_handle_t handle,
                       const hw_buffer_desc_t *input,
                       const char *context_json,
                       infer_result_t *result);

    /** @brief Destroy algorithm context */
    void detector_destroy(algo_handle_t handle);

    /** @brief Get algorithm version */
    const char *detector_version(void);

    /** @brief Get algorithm name */
    const char *detector_name(void);

    /** @brief Run self-test */
    int detector_self_test(void);

    /** @brief Update face library (hot-swap) */
    int detector_update_face_library(algo_handle_t handle, const char *face_library_json);

    /** @brief Free inference result memory */
    void algo_free_result(infer_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // GPU_FACE_RECOGNITION_ALGO_API_H
