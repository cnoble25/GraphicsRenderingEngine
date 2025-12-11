#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include "cuda_utils.h"

// RAII wrapper for CUDA device memory
// Automatically manages CUDA memory allocation and deallocation
// Exception-safe: guarantees no memory leaks even if exceptions are thrown
template<typename T>
class CudaBuffer {
private:
    T* ptr_;
    size_t count_;
    
    // Helper to check CUDA errors with context
    void check_error(cudaError_t error, const char* operation) {
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("CudaBuffer ") + operation + " failed: " + 
                cudaGetErrorString(error)
            );
        }
    }
    
public:
    // Construct buffer with specified count of elements
    explicit CudaBuffer(size_t count) : count_(count), ptr_(nullptr) {
        if (count > 0) {
            cudaError_t err = cudaMalloc((void**)&ptr_, count * sizeof(T));
            check_error(err, "cudaMalloc");
        }
    }
    
    // Destructor - automatically frees memory
    // noexcept ensures this never throws, maintaining exception safety
    ~CudaBuffer() noexcept {
        if (ptr_) {
            // Ignore errors in destructor (can't throw from destructor)
            // This is safe because:
            // 1. If cudaFree fails, it's likely a serious system error
            // 2. We can't throw from destructor anyway
            // 3. The memory will be cleaned up by CUDA runtime on exit
            cudaFree(ptr_);
            ptr_ = nullptr;  // Prevent double-free if destructor called multiple times
        }
    }
    
    // Delete copy constructor and assignment (no copying)
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Move constructor
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    // Move assignment
    // noexcept ensures this never throws, maintaining exception safety
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            // Free existing memory before taking ownership of new memory
            // This ensures no memory leaks even if move assignment throws
            if (ptr_) {
                cudaFree(ptr_);
                ptr_ = nullptr;
            }
            // Transfer ownership
            ptr_ = other.ptr_;
            count_ = other.count_;
            // Clear source to prevent double-free
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // Get raw pointer (for CUDA kernel calls)
    T* get() const { return ptr_; }
    
    // Get number of elements
    size_t size() const { return count_; }
    
    // Get size in bytes
    size_t bytes() const { return count_ * sizeof(T); }
    
    // Copy data from host to device
    // Strong exception guarantee: if copy fails, buffer state is unchanged
    void copy_from_host(const T* host_data, size_t count) {
        if (!ptr_) {
            throw std::runtime_error("CudaBuffer: Cannot copy to invalid buffer");
        }
        if (host_data == nullptr && count > 0) {
            throw std::invalid_argument("CudaBuffer: host_data cannot be null when count > 0");
        }
        if (count > count_) {
            count = count_;
        }
        if (count > 0) {
            cudaError_t err = cudaMemcpy(ptr_, host_data, count * sizeof(T), 
                                         cudaMemcpyHostToDevice);
            check_error(err, "copy_from_host");
        }
    }
    
    // Copy data from device to host
    // Strong exception guarantee: if copy fails, buffer state is unchanged
    void copy_to_host(T* host_data, size_t count) const {
        if (!ptr_) {
            throw std::runtime_error("CudaBuffer: Cannot copy from invalid buffer");
        }
        if (host_data == nullptr && count > 0) {
            throw std::invalid_argument("CudaBuffer: host_data cannot be null when count > 0");
        }
        if (count > count_) {
            count = count_;
        }
        if (count > 0) {
            cudaError_t err = cudaMemcpy(host_data, ptr_, count * sizeof(T), 
                                         cudaMemcpyDeviceToHost);
            check_error(err, "copy_to_host");
        }
    }
    
    // Check if buffer is valid (has allocated memory)
    bool valid() const { return ptr_ != nullptr; }
};

#endif // CUDA_MEMORY_H
