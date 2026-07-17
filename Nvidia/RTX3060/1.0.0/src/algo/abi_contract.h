/**
 * @file abi_contract.h
 * @brief GPU Face Recognition — C ABI contract specification.
 *        All .so must export these symbols with outermost try-catch.
 *
 * Required symbols: detector_init, detector_infer, detector_destroy
 * Optional symbols: detector_version, detector_name, detector_self_test,
 *                   detector_update_face_library, algo_free_result
 *
 * Version: 1.0.0
 */

#ifndef AIVISION_ALGO_ABI_CONTRACT_H
#define AIVISION_ALGO_ABI_CONTRACT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // ============================================================
    // Type Definitions
    // ============================================================

    /// Algorithm context handle (opaque pointer)
    typedef struct algo_context_t *algo_handle_t;

    /// Hardware buffer descriptor (engine ABI compatible)
    typedef struct
    {
        // ---- Engine ABI fields (first 56 bytes) ----
        int dma_fd;
        size_t size;
        uint32_t width;
        uint32_t height;
        uint32_t pixel_format;
        int dma_buf_fd;
        uint64_t phys_addr;
        void *data;
        uint32_t stride;

        // ---- Package extension fields ----
        uint32_t buffer_type;
        uint32_t buffer_owner;
        uint32_t reserved_flags;

        /// Platform-dependent union (80 bytes)
        union {
            uint8_t plat[80];
            uint64_t plat_u64[10];
        };

    } hw_buffer_desc_t; // sizeof = 144

    /// Inference result
    typedef struct
    {
        char *result_json;
        size_t result_json_len;
        uint32_t infer_time_us;
        int reserved[4];
    } infer_result_t; // sizeof = 40

    // ============================================================
    // Required Symbols (Must Export)
    // ============================================================

    /// Initialize algorithm context.
    algo_handle_t detector_init(const char *config_json);

    /// Run inference on one frame.
    int detector_infer(algo_handle_t handle,
                       const hw_buffer_desc_t *input,
                       const char *context_json,
                       infer_result_t *result);

    /// Destroy algorithm context.
    void detector_destroy(algo_handle_t handle);

    // ============================================================
    // Optional Symbols
    // ============================================================

    /// Get algorithm version.
    const char *detector_version(void);

    /// Get algorithm name.
    const char *detector_name(void);

    /// Run self-test.
    int detector_self_test(void);

    /// Hot-update face library.
    int detector_update_face_library(algo_handle_t handle, const char *face_library_json);

    /// Free inference result memory.
    void algo_free_result(infer_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // AIVISION_ALGO_ABI_CONTRACT_H
