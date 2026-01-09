#ifndef ERRORS_H
#define ERRORS_H

// Error codes for renderer operations
// These match the C API return convention where 0 = failure, non-zero = success
// But we use enum values to make errors more descriptive
enum class RendererError {
    SUCCESS = 1,                    // Operation succeeded
    INVALID_SCENE_HANDLE = 2,       // Null or invalid scene handle
    INVALID_OBJECT = 3,             // Null or invalid object pointer
    INVALID_INDEX = 4,              // Index out of bounds
    OBJ_FILE_NOT_FOUND = 5,         // OBJ file path not found
    OBJ_FILE_LOAD_FAILED = 6,       // Failed to load OBJ file
    CUDA_DEVICE_INIT_FAILED = 7,    // CUDA device initialization failed
    CUDA_MALLOC_FAILED = 8,         // CUDA memory allocation failed
    CUDA_MEMCPY_FAILED = 9,         // CUDA memory copy failed
    CUDA_KERNEL_LAUNCH_FAILED = 10, // CUDA kernel launch failed
    FILE_WRITE_FAILED = 11,         // Failed to write output file
    EMPTY_SCENE = 12,               // Scene has no objects
    NO_TRIANGLES_IN_SCENE = 13      // Scene has no triangles
};

// Get human-readable error message
const char* get_error_message(RendererError error);

#endif // ERRORS_H
