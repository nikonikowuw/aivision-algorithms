/**
 * @file cuda_utils.h
 * @brief GPU Face Recognition — CUDA memory and stream management utilities.
 * @module Runtime Layer
 */

#ifndef GPU_FACE_RECOGNITION_CUDA_UTILS_H
#define GPU_FACE_RECOGNITION_CUDA_UTILS_H

#include <cstddef>
#include <cstdint>

// Forward-declare CUDA types to avoid requiring CUDA headers at include time
struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace face_rec {

/**
 * @class CudaStream
 * @brief RAII wrapper for a CUDA stream.
 */
class CudaStream {
public:
    CudaStream();
    ~CudaStream();

    // Non-copyable, movable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    /** @brief Get the underlying CUDA stream handle */
    cudaStream_t Get() const;

    /** @brief Synchronize the stream (wait for all pending work) */
    bool Synchronize();

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @class CudaDeviceBuffer
 * @brief RAII wrapper for GPU device memory allocation.
 */
class CudaDeviceBuffer {
public:
    explicit CudaDeviceBuffer(size_t bytes = 0);
    ~CudaDeviceBuffer();

    // Non-copyable, movable
    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

    /** @brief Allocate or reallocate device memory */
    bool Allocate(size_t bytes);

    /** @brief Free device memory */
    void Free();

    /** @brief Get device pointer */
    void* Get() const;

    /** @brief Get allocated size in bytes */
    size_t Size() const;

    /** @brief Check if buffer is valid (allocated) */
    bool IsValid() const;

private:
    void* device_ptr_ = nullptr;
    size_t size_ = 0;
};

/**
 * @brief Copy data from host to device.
 * @param dst Device pointer
 * @param src Host pointer
 * @param bytes Number of bytes to copy
 * @param stream CUDA stream (0 for default)
 * @return true on success
 */
bool CudaMemcpyH2D(void* dst, const void* src, size_t bytes, cudaStream_t stream = 0);

/**
 * @brief Copy data from device to host.
 * @param dst Host pointer
 * @param src Device pointer
 * @param bytes Number of bytes to copy
 * @param stream CUDA stream (0 for default)
 * @return true on success
 */
bool CudaMemcpyD2H(void* dst, const void* src, size_t bytes, cudaStream_t stream = 0);

/**
 * @brief Set the CUDA device for the current thread.
 * @param device_id CUDA device ID
 * @return true on success
 */
bool CudaSetDevice(int device_id);

/**
 * @brief Get the current CUDA device ID.
 * @return CUDA device ID, or -1 on error
 */
int CudaGetDevice();

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_CUDA_UTILS_H
