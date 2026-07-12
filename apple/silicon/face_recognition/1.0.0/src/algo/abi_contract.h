#ifndef AIVISION_ALGO_ABI_CONTRACT_H
#define AIVISION_ALGO_ABI_CONTRACT_H

// ABI 契约规范 — 所有算法 .so 必须遵守的 C ABI。
//
// 必需符号 (必须导出)：
//   detector_init      — 初始化算法上下文
//   detector_infer     — 对一帧执行推理
//   detector_destroy   — 销毁算法上下文
//
// 可选符号 (可导出用于元信息)：
//   detector_version   — 算法版本号
//   detector_name      — 算法名称
//   detector_self_test — 自检函数
//   detector_update_face_library — 热更新算法实例内存人脸库快照
//
// 版本: 1.0.0

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // ============================================================
    // 类型定义
    // ============================================================

    /// 算法上下文句柄 (不透明指针)
    typedef struct algo_context_t *algo_handle_t;

    /// 硬件缓冲区描述符 (扩展版，包含平台联合体)
    /// 前 56 字节与引擎 ABI 完全兼容，后接算法包扩展字段。
    typedef struct
    {
        // ---- 引擎 ABI 兼容字段 (前 56 字节) ----
        int dma_fd;            // DMA 文件描述符
        size_t size;           // 缓冲区大小
        uint32_t width;        // 图像宽度
        uint32_t height;       // 图像高度
        uint32_t pixel_format; // 像素格式 (FourCC)
        int dma_buf_fd;        // DMA-BUF 文件描述符 (可选)
        uint64_t phys_addr;    // 物理地址 (可选)
        void *data;            // HostMemory 数据指针
        uint32_t stride;       // 每行字节数

        // ---- 算法包扩展字段 ----
        uint32_t buffer_type;      // 缓冲区类型 (0=未知, 1=CPU, 2=DMA-BUF)
        uint32_t buffer_owner;     // 缓冲区所有者标识
        uint32_t reserved_flags;   // 预留标志位

        /// 平台相关联合体 (80 字节)
        union {
            uint8_t plat[80];
            uint64_t plat_u64[10];
        };

    } hw_buffer_desc_t; // sizeof = 144

    /// 推理结果
    typedef struct
    {
        /// 结果 JSON 字符串 (由算法内部 malloc，由调用者 free)
        char *result_json;

        /// 结果 JSON 长度
        size_t result_json_len;

        /// 推理耗时 (微秒)
        uint32_t infer_time_us;

        /// 预留字段
        int reserved[4];
    } infer_result_t; // sizeof = 40

    // ============================================================
    // 必需符号 (Must Export)
    // ============================================================

    /// 初始化算法上下文。
    /// @param config_json 算法配置参数的 JSON 字符串 (由调用者管理生命周期)
    /// @return 算法句柄，失败返回 NULL
    algo_handle_t detector_init(const char *config_json);

    /// 对一帧执行推理。
    /// @param handle  算法句柄 (由 detector_init 返回)
    /// @param input   输入帧的硬件缓冲区描述符 (零拷贝)
    /// @param context_json  算法串联上下文 JSON (上一级算法输出，可为 NULL)
    /// @param result  输出推理结果
    /// @return 0 成功，非 0 失败
    int detector_infer(algo_handle_t handle,
                       const hw_buffer_desc_t *input,
                       const char *context_json,
                       infer_result_t *result);

    /// 销毁算法上下文，释放所有资源。
    /// @param handle 算法句柄
    void detector_destroy(algo_handle_t handle);

    // ============================================================
    // 可选符号 (Optional)
    // ============================================================

    /// 获取算法版本号。
    /// @return 版本字符串 (如 "1.2.0")，由算法内部管理生命周期
    const char *detector_version(void);

    /// 获取算法名称。
    /// @return 名称字符串 (如 "person_detector")，由算法内部管理生命周期
    const char *detector_name(void);

    /// 执行自检。
    /// @return 0 自检通过，非 0 自检失败
    int detector_self_test(void);

    /// 热更新算法实例内存人脸库快照。
    /// 该接口为可选符号。Engine 只转发完整快照 JSON，不解释新增/删除等业务语义。
    /// @param handle 算法句柄
    /// @param face_library_json 完整人脸库快照 JSON，由调用者管理生命周期
    /// @return 0 更新成功，非 0 更新失败或算法不支持
    int detector_update_face_library(algo_handle_t handle, const char *face_library_json);

    // ============================================================
    // 辅助函数 (由引擎提供，算法可调用)
    // ============================================================

    /// 释放推理结果中的 JSON 字符串。
    /// 当调用者使用完 result_json 后，调用此函数释放内存。
    void algo_free_result(infer_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // AIVISION_ALGO_ABI_CONTRACT_H
