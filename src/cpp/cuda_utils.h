#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace cuda_utils {

/**
 * Check CUDA error and throw exception if error occurred
 * @param error CUDA error code to check
 * @param file Source file name (use __FILE__)
 * @param line Line number (use __LINE__)
 * @param operation Description of the operation that failed
 * @throws std::runtime_error if error != cudaSuccess
 */
inline void check_cuda_error(cudaError_t error, const char* file, int line, const char* operation) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line 
            << " in " << operation 
            << ": " << cudaGetErrorString(error) 
            << " (error code: " << static_cast<int>(error) << ")";
        throw std::runtime_error(oss.str());
    }
}

/**
 * Macro to check CUDA calls with automatic file/line information
 * Usage: CHECK_CUDA(cudaMalloc(...));
 */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t _cuda_error = (call); \
        cuda_utils::check_cuda_error(_cuda_error, __FILE__, __LINE__, #call); \
    } while(0)

/**
 * Synchronize CUDA device and check for errors
 * @throws std::runtime_error if synchronization fails
 */
inline void synchronize_device() {
    CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * Get last CUDA error and throw if error occurred
 * Useful after kernel launches
 * @param operation Description of the operation
 * @throws std::runtime_error if error occurred
 */
inline void check_last_error(const char* operation) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error after " << operation 
            << ": " << cudaGetErrorString(error) 
            << " (error code: " << static_cast<int>(error) << ")";
        throw std::runtime_error(oss.str());
    }
}

/**
 * Reset CUDA device (useful for cleanup)
 * @throws std::runtime_error if reset fails
 */
inline void reset_device() {
    CHECK_CUDA(cudaDeviceReset());
}

/**
 * Set CUDA device
 * @param device Device ID to set
 * @throws std::runtime_error if device cannot be set
 */
inline void set_device(int device) {
    CHECK_CUDA(cudaSetDevice(device));
}

} // namespace cuda_utils

#endif // CUDA_UTILS_H
