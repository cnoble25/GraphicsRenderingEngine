#include "errors.h"

const char* get_error_message(RendererError error) {
    switch (error) {
        case RendererError::SUCCESS:
            return "Success";
        case RendererError::INVALID_SCENE_HANDLE:
            return "Invalid scene handle";
        case RendererError::INVALID_OBJECT:
            return "Invalid object";
        case RendererError::INVALID_INDEX:
            return "Invalid index";
        case RendererError::OBJ_FILE_NOT_FOUND:
            return "OBJ file not found";
        case RendererError::OBJ_FILE_LOAD_FAILED:
            return "Failed to load OBJ file";
        case RendererError::CUDA_DEVICE_INIT_FAILED:
            return "CUDA device initialization failed";
        case RendererError::CUDA_MALLOC_FAILED:
            return "CUDA memory allocation failed";
        case RendererError::CUDA_MEMCPY_FAILED:
            return "CUDA memory copy failed";
        case RendererError::CUDA_KERNEL_LAUNCH_FAILED:
            return "CUDA kernel launch failed";
        case RendererError::FILE_WRITE_FAILED:
            return "File write failed";
        case RendererError::EMPTY_SCENE:
            return "Scene is empty";
        case RendererError::NO_TRIANGLES_IN_SCENE:
            return "No triangles in scene";
        default:
            return "Unknown error";
    }
}
