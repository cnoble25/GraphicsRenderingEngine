cmake_minimum_required(VERSION 3.18)
project(GraphicsRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add executable with CUDA support
add_executable(GraphicsRenderer 
        main.cpp
        ray_trace_cuda.cu
        ray_trace_cuda_helper.cpp
        vec3.h
        color.h
        ray.h
        vertex.h
        model.h
        transform.h
        rotation.h
        ray_trace_cuda.h
)

# Enable CUDA for this target
set_property(TARGET GraphicsRenderer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture (adjust based on your GPU)
# CUDA 11.5 supports: 50, 60, 61, 70, 75, 80, 86
# Leave empty to auto-detect, or specify your GPU's compute capability
set_property(TARGET GraphicsRenderer PROPERTY CUDA_ARCHITECTURES "50;60;70;75;80;86")

# Add shared library for .NET interop
add_library(GraphicsRendererAPI SHARED
    renderer_api.cpp
    obj_loader.cpp
    ray_trace_cuda.cu
    ray_trace_cuda_helper.cpp
    vec3.h
    color.h
    ray.h
    vertex.h
    model.h
    transform.h
    rotation.h
    ray_trace_cuda.h
    renderer_api.h
    obj_loader.h
)

# Enable CUDA for the API library
set_property(TARGET GraphicsRendererAPI PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GraphicsRendererAPI PROPERTY CUDA_ARCHITECTURES "50;60;70;75;80;86")

# Set output directory for the shared library
set_target_properties(GraphicsRendererAPI PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
