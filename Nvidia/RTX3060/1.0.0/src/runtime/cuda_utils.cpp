/**
 * @file cuda_utils.cpp
 * @brief GPU Face Recognition — CUDA memory and stream management implementation.
 */

#include "cuda_utils.h"
#include "common/logger.h"
#include <cuda_runtime.h>

namespace face_rec {

// ---- CudaStream ----

CudaStream::CudaStream() {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaStreamCreate failed: %s", cudaGetErrorString(err));
        stream_ = nullptr;
    }
}

CudaStream::~CudaStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (this != &other) {
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

cudaStream_t CudaStream::Get() const {
    return stream_;
}

bool CudaStream::Synchronize() {
    if (!stream_) return false;
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ---- CudaDeviceBuffer ----

CudaDeviceBuffer::CudaDeviceBuffer(size_t bytes) {
    if (bytes > 0) {
        Allocate(bytes);
    }
}

CudaDeviceBuffer::~CudaDeviceBuffer() {
    Free();
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_) {
    other.device_ptr_ = nullptr;
    other.size_ = 0;
}

CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
    if (this != &other) {
        Free();
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        other.device_ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

bool CudaDeviceBuffer::Allocate(size_t bytes) {
    Free();
    cudaError_t err = cudaMalloc(&device_ptr_, bytes);
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaMalloc(%zu) failed: %s", bytes, cudaGetErrorString(err));
        device_ptr_ = nullptr;
        size_ = 0;
        return false;
    }
    size_ = bytes;
    return true;
}

void CudaDeviceBuffer::Free() {
    if (device_ptr_) {
        cudaFree(device_ptr_);
        device_ptr_ = nullptr;
        size_ = 0;
    }
}

void* CudaDeviceBuffer::Get() const {
    return device_ptr_;
}

size_t CudaDeviceBuffer::Size() const {
    return size_;
}

bool CudaDeviceBuffer::IsValid() const {
    return device_ptr_ != nullptr && size_ > 0;
}

// ---- Memory copy helpers ----

bool CudaMemcpyH2D(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaMemcpy H2D failed: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaMemcpyD2H(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err;
    if (stream) {
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    } else {
        err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaMemcpy D2H failed: %s", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool CudaSetDevice(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaSetDevice(%d) failed: %s", device_id, cudaGetErrorString(err));
        return false;
    }
    return true;
}

int CudaGetDevice() {
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        ALGO_LOGE(BACKEND, "cudaGetDevice failed: %s", cudaGetErrorString(err));
        return -1;
    }
    return device;
}

} // namespace face_rec
