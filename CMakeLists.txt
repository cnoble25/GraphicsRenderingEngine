cmake_minimum_required(VERSION 3.18)
project(GraphicsRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/cuda)

# Add executable with CUDA support
add_executable(GraphicsRenderer 
        src/cpp/main.cpp
        src/cpp/jpg_writer.cpp
        src/cuda/ray_trace_cuda.cu
        src/cuda/ray_trace_cuda_helper.cpp
        src/cpp/vec3.h
        src/cpp/color.h
        src/cpp/stb_image_write.h
        src/cpp/ray.h
        src/cpp/vertex.h
        src/cpp/model.h
        src/cpp/transform.h
        src/cpp/rotation.h
        src/cpp/cuda_memory.h
        src/cpp/cuda_utils.h
        src/cpp/camera_setup.h
        src/cpp/constants.h
        src/cpp/strong_types.h
        src/cuda/ray_trace_cuda.h
)

# Link SDL2 to executable
target_link_libraries(GraphicsRenderer ${SDL2_LIBRARIES})

# Enable CUDA for this target
set_property(TARGET GraphicsRenderer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture (adjust based on your GPU)
# CUDA 11.5 supports: 50, 60, 61, 70, 75, 80, 86
# Leave empty to auto-detect, or specify your GPU's compute capability
set_property(TARGET GraphicsRenderer PROPERTY CUDA_ARCHITECTURES "50;60;70;75;80;86")

# Add shared library for .NET interop
add_library(GraphicsRendererAPI SHARED
    src/cpp/renderer_api.cpp
    src/cpp/jpg_writer.cpp
    src/cpp/obj_loader.cpp
    src/cpp/errors.cpp
    src/cuda/ray_trace_cuda.cu
    src/cuda/ray_trace_cuda_helper.cpp
    src/cpp/vec3.h
    src/cpp/color.h
    src/cpp/stb_image_write.h
    src/cpp/ray.h
    src/cpp/vertex.h
    src/cpp/model.h
    src/cpp/transform.h
    src/cpp/rotation.h
    src/cpp/light.h
    src/cpp/errors.h
    src/cpp/cuda_memory.h
    src/cpp/cuda_utils.h
    src/cpp/camera_setup.h
    src/cpp/constants.h
    src/cpp/strong_types.h
    src/cuda/ray_trace_cuda.h
    src/cpp/renderer_api.h
    src/cpp/obj_loader.h
)

# Link SDL2 to API library
target_link_libraries(GraphicsRendererAPI ${SDL2_LIBRARIES})

# Define export macro for Windows
if(WIN32)
    target_compile_definitions(GraphicsRendererAPI PRIVATE GRAPHICS_RENDERER_API_EXPORTS)
endif()

# Enable CUDA for the API library
set_property(TARGET GraphicsRendererAPI PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GraphicsRendererAPI PROPERTY CUDA_ARCHITECTURES "50;60;70;75;80;86")

# Set output directory for the shared library
if(WIN32)
    # Windows: DLL goes to RUNTIME_OUTPUT_DIRECTORY
    set_target_properties(GraphicsRendererAPI PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
    )
else()
    # Linux/macOS: Shared library goes to LIBRARY_OUTPUT_DIRECTORY
    set_target_properties(GraphicsRendererAPI PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

# Add PPM to JPG converter utility (no CUDA needed)
add_executable(ppm_to_jpg
    src/cpp/ppm_to_jpg.cpp
    src/cpp/jpg_writer.cpp
    src/cpp/vec3.h
    src/cpp/color.h
    src/cpp/stb_image_write.h
)
