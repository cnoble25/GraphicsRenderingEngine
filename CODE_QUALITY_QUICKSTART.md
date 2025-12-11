# Code Quality Improvements - Quick Start Guide

This is a condensed guide to get started with code quality improvements immediately.

## Priority Order

### 🔴 Critical (Do First)
1. **CUDA Memory RAII** - Prevents memory leaks, enables exception safety
2. **Error Handling** - Better debugging and user experience
3. **Remove goto statements** - Modern C++ best practice

### 🟡 High Priority (Do Next)
4. **Const Correctness** - Prevents bugs, improves code clarity
5. **Code Formatting** - Consistency, easier reviews
6. **Fix Naming Issues** - `pyamid()` typo, inconsistent names

### 🟢 Medium Priority (Do After)
7. **Model Update Efficiency** - Performance improvement
8. **Type Safety** - Strong types, fewer bugs
9. **Code Duplication** - DRY principle

## Quick Start: Phase 1 (Critical)

### Step 1: Create CUDA Memory Wrapper (30 minutes)

Create `src/cpp/cuda_memory.h`:

```cpp
#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

template<typename T>
class CudaBuffer {
private:
    T* ptr_;
    size_t count_;
    
    void check_error(cudaError_t error, const char* operation) {
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + 
                cudaGetErrorString(error)
            );
        }
    }
    
public:
    explicit CudaBuffer(size_t count) : count_(count), ptr_(nullptr) {
        if (count > 0) {
            cudaError_t err = cudaMalloc((void**)&ptr_, count * sizeof(T));
            check_error(err, "cudaMalloc");
        }
    }
    
    ~CudaBuffer() noexcept {
        if (ptr_) {
            cudaFree(ptr_);  // Ignore errors in destructor
        }
    }
    
    // Delete copy
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    
    // Move
    CudaBuffer(CudaBuffer&& other) noexcept 
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    
    void copy_from_host(const T* host_data, size_t count) {
        if (count > count_) count = count_;
        cudaError_t err = cudaMemcpy(ptr_, host_data, count * sizeof(T), 
                                     cudaMemcpyHostToDevice);
        check_error(err, "cudaMemcpy HostToDevice");
    }
    
    void copy_to_host(T* host_data, size_t count) const {
        if (count > count_) count = count_;
        cudaError_t err = cudaMemcpy(host_data, ptr_, count * sizeof(T), 
                                     cudaMemcpyDeviceToHost);
        check_error(err, "cudaMemcpy DeviceToHost");
    }
};

#endif // CUDA_MEMORY_H
```

### Step 2: Create Error Codes (15 minutes)

Create `src/cpp/errors.h`:

```cpp
#ifndef ERRORS_H
#define ERRORS_H

enum class RendererError {
    SUCCESS = 0,
    INVALID_SCENE_HANDLE = 1,
    INVALID_OBJECT = 2,
    INVALID_INDEX = 3,
    OBJ_FILE_NOT_FOUND = 4,
    OBJ_FILE_LOAD_FAILED = 5,
    CUDA_DEVICE_INIT_FAILED = 6,
    CUDA_MALLOC_FAILED = 7,
    CUDA_MEMCPY_FAILED = 8,
    CUDA_KERNEL_LAUNCH_FAILED = 9,
    FILE_WRITE_FAILED = 10,
    EMPTY_SCENE = 11,
    NO_TRIANGLES_IN_SCENE = 12
};

const char* get_error_message(RendererError error);

#endif // ERRORS_H
```

Create `src/cpp/errors.cpp`:

```cpp
#include "errors.h"

const char* get_error_message(RendererError error) {
    switch (error) {
        case RendererError::SUCCESS: return "Success";
        case RendererError::INVALID_SCENE_HANDLE: return "Invalid scene handle";
        case RendererError::INVALID_OBJECT: return "Invalid object";
        case RendererError::INVALID_INDEX: return "Invalid index";
        case RendererError::OBJ_FILE_NOT_FOUND: return "OBJ file not found";
        case RendererError::OBJ_FILE_LOAD_FAILED: return "Failed to load OBJ file";
        case RendererError::CUDA_DEVICE_INIT_FAILED: return "CUDA device initialization failed";
        case RendererError::CUDA_MALLOC_FAILED: return "CUDA memory allocation failed";
        case RendererError::CUDA_MEMCPY_FAILED: return "CUDA memory copy failed";
        case RendererError::CUDA_KERNEL_LAUNCH_FAILED: return "CUDA kernel launch failed";
        case RendererError::FILE_WRITE_FAILED: return "File write failed";
        case RendererError::EMPTY_SCENE: return "Scene is empty";
        case RendererError::NO_TRIANGLES_IN_SCENE: return "No triangles in scene";
        default: return "Unknown error";
    }
}
```

### Step 3: Refactor One Function (1 hour)

Refactor `render_scene()` in `renderer_api.cpp`:

**Before:**
```cpp
Vertex_cuda* d_models = nullptr;
int* d_triangle_counts = nullptr;
// ... more pointers
cudaStatus = cudaMalloc(...);
if (cudaStatus != cudaSuccess) {
    goto cleanup;
}
cleanup:
    if (d_models) cudaFree(d_models);
```

**After:**
```cpp
try {
    CudaBuffer<Vertex_cuda> d_models(all_triangles.size());
    CudaBuffer<int> d_triangle_counts(triangle_counts.size());
    CudaBuffer<int> d_triangle_offsets(triangle_offsets.size());
    CudaBuffer<Color_cuda> d_image(image_width * image_height);
    
    d_models.copy_from_host(all_triangles.data(), all_triangles.size());
    // ... rest of code
    // No cleanup needed - automatic!
} catch (const std::exception& e) {
    std::cerr << "Rendering failed: " << e.what() << std::endl;
    return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
}
```

### Step 4: Update CMakeLists.txt (5 minutes)

Add new files to CMakeLists.txt:
```cmake
add_library(GraphicsRendererAPI SHARED
    # ... existing files ...
    src/cpp/errors.cpp
    src/cpp/cuda_memory.h  # Header-only, but list for clarity
)
```

## Testing After Phase 1

1. **Build**: `cd build-ui && cmake .. && make`
2. **Run Tests**: Execute existing test programs
3. **Check Memory**: Use `valgrind` or CUDA memory checker
4. **Verify Output**: Ensure renders are identical

## Next Steps

After Phase 1 is complete and tested:
1. Move to Phase 2 (Memory Management improvements)
2. Then Phase 3 (Code Organization)
3. Continue with remaining phases

## Common Issues & Solutions

### Issue: CUDA errors not being caught
**Solution**: Ensure CUDA errors are checked immediately after calls

### Issue: Performance regression
**Solution**: Profile before/after. RAII should have zero overhead.

### Issue: Compilation errors
**Solution**: Ensure all headers are included, check C++17 standard is set

## Tips

- **One file at a time**: Don't refactor everything at once
- **Test frequently**: After each function refactor, test
- **Use git**: Commit after each successful refactor
- **Keep old code**: Comment out old code initially, remove after testing

## Time Estimates

- **CudaBuffer class**: 30-45 minutes
- **Error codes**: 15-30 minutes  
- **Refactor render_scene()**: 1-2 hours
- **Testing**: 30 minutes
- **Total**: ~3-4 hours for Phase 1 foundation
