#ifndef AIVISION_ALGO_SUBMODULE_ABI_CONTRACT_H
#define AIVISION_ALGO_SUBMODULE_ABI_CONTRACT_H

// This is the submodule copy of engine/include/algo/abi_contract.h.
// Must have identical struct layout for ABI compatibility with the engine.

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct algo_context_t *algo_handle_t;

// ============================================================
// v2: Buffer type discriminator and platform-specific structs
// ============================================================

typedef enum {
    HW_BUFFER_TYPE_DEFAULT       = 0,
    HW_BUFFER_TYPE_ASCEND_DVPP   = 1,
    HW_BUFFER_TYPE_ASCEND_DEVICE = 2,
    HW_BUFFER_TYPE_ROCKCHIP_MPP  = 3,
    HW_BUFFER_TYPE_ROCKCHIP_RGA  = 4,
    HW_BUFFER_TYPE_CUDA          = 5,
    HW_BUFFER_TYPE_APPLE_NATIVE  = 6,
} hw_buffer_type_t;

typedef enum {
    HW_BUFFER_OWNER_ENGINE    = 0,
    HW_BUFFER_OWNER_ALGORITHM = 1,
} hw_buffer_owner_t;

typedef enum {
    HW_BUFFER_APPLE_NONE          = 0,
    HW_BUFFER_APPLE_CVPIXELBUFFER = 1,
    HW_BUFFER_APPLE_IOSURFACE     = 2,
} hw_buffer_apple_kind_t;

#define HW_BUFFER_APPLE_ABI_VERSION 1
#define HW_BUFFER_PLANE_COUNT_UNKNOWN 0

typedef struct {
    int32_t  device_id;
    int32_t  memory_subtype;
    uint32_t pixel_format;
    uint32_t aligned_width;
    uint32_t aligned_height;
    uint32_t alignment;
    int32_t  cache_synced;
    int32_t  color_space;
    uint32_t reserved_flags;
    void    *data_ptr;
    uint32_t reserved_pad[3];
} hw_buffer_ascend_t;

typedef struct {
    int32_t  buffer_subtype;
    int32_t  hor_stride;
    int32_t  ver_stride;
    uint32_t reserved_pad[13];
} hw_buffer_rockchip_t;

typedef struct { uint32_t reserved_pad[16]; } hw_buffer_cuda_t;

// Apple CoreVideo / IOSurface 原生缓冲区载荷。
// native_handle 持有 Engine 借出的 retained CVPixelBufferRef (或 IOSurfaceRef)，
// 算法在同步 detector_infer 调用期间 borrow，不可 release 或存储到调用之外。
typedef struct {
    uint16_t abi_version;   // HW_BUFFER_APPLE_ABI_VERSION
    uint16_t struct_size;   // sizeof(hw_buffer_apple_t)
    uint32_t buffer_kind;   // hw_buffer_apple_kind_t
    uint32_t pixel_format;  // CoreVideo / IOSurface FourCC
    uint32_t plane_count;   // 1 (BGRA) 或 2 (NV12)，0=由算法自行查询
    uint64_t native_handle; // retained CVPixelBufferRef / IOSurfaceRef
    uint32_t plane_stride[2];
    uint32_t plane_offset[2];
    uint64_t synchronization; // 保留同步 token
    uint32_t reserved_pad[3]; // 填充至 64 字节
} hw_buffer_apple_t;

typedef struct {
    // v1 fields (ABI compat, must match engine side exactly)
    int       dma_fd;
    size_t    size;
    uint32_t  width;
    uint32_t  height;
    uint32_t  pixel_format;
    int       dma_buf_fd;
    uint64_t  phys_addr;
    void     *data;
    uint32_t  stride;

    // v2 fields
    int32_t  buffer_type;
    int32_t  buffer_owner;
    uint32_t reserved_flags;

    union {
        hw_buffer_ascend_t   ascend;
        hw_buffer_rockchip_t rockchip;
        hw_buffer_cuda_t     cuda;
        hw_buffer_apple_t    apple;
    } plat;

    int64_t  reserved_padding[2];
} hw_buffer_desc_t;

#if defined(__cplusplus)
static_assert(sizeof(hw_buffer_ascend_t) == 64, "hw_buffer_ascend_t must be 64 bytes");
static_assert(sizeof(hw_buffer_rockchip_t) == 64, "hw_buffer_rockchip_t must be 64 bytes");
static_assert(sizeof(hw_buffer_cuda_t) == 64, "hw_buffer_cuda_t must be 64 bytes");
static_assert(sizeof(hw_buffer_apple_t) == 64, "hw_buffer_apple_t must be 64 bytes");
static_assert(sizeof(hw_buffer_desc_t) == 144, "hw_buffer_desc_t must be 144 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(hw_buffer_ascend_t) == 64, "hw_buffer_ascend_t must be 64 bytes");
_Static_assert(sizeof(hw_buffer_rockchip_t) == 64, "hw_buffer_rockchip_t must be 64 bytes");
_Static_assert(sizeof(hw_buffer_cuda_t) == 64, "hw_buffer_cuda_t must be 64 bytes");
_Static_assert(sizeof(hw_buffer_apple_t) == 64, "hw_buffer_apple_t must be 64 bytes");
_Static_assert(sizeof(hw_buffer_desc_t) == 144, "hw_buffer_desc_t must be 144 bytes");
#endif

typedef struct {
    char   *result_json;
    size_t  result_json_len;
    uint32_t infer_time_us;
    int     reserved[4];
} infer_result_t;

// Required symbols
algo_handle_t detector_init(const char *config_json);
int detector_infer(algo_handle_t handle, const hw_buffer_desc_t *input,
                   const char *context_json, infer_result_t *result);
void detector_destroy(algo_handle_t handle);

// Optional symbols
const char *detector_version(void);
const char *detector_name(void);
int detector_self_test(void);
int detector_update_face_library(algo_handle_t handle, const char *face_library_json);

// Helper
void algo_free_result(infer_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // AIVISION_ALGO_SUBMODULE_ABI_CONTRACT_H
