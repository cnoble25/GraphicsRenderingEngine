# Graphics Rendering Engine

A CUDA-accelerated 3D graphics rendering engine with ray tracing capabilities and a .NET WPF UI interface.

## Project Overview

This project is a GPU-accelerated ray tracing renderer that uses CUDA for parallel computation. It includes a command-line renderer and a .NET-based graphical user interface for managing 3D scenes.

## Source Files Location

### Expected Directory Structure

The source files for this project should be organized in the following structure:

```
GraphicsRenderingEngine/
├── src/
│   ├── cpp/                           # C++ source files
│   │   ├── main.cpp                   # Main executable entry point
│   │   ├── renderer_api.cpp           # C API for .NET interop
│   │   ├── renderer_api.h             # API header
│   │   ├── jpg_writer.cpp             # JPEG writing utilities
│   │   ├── obj_loader.cpp             # OBJ file loader implementation
│   │   ├── obj_loader.h               # OBJ loader header
│   │   ├── errors.cpp                 # Error handling
│   │   ├── errors.h                   # Error definitions
│   │   ├── ppm_to_jpg.cpp             # PPM to JPEG converter utility
│   │   ├── vec3.h                     # 3D vector math
│   │   ├── color.h                    # Color utilities
│   │   ├── ray.h                      # Ray structure
│   │   ├── vertex.h                   # Vertex structure
│   │   ├── model.h                    # 3D model structure
│   │   ├── transform.h                # Transformation utilities
│   │   ├── rotation.h                 # Rotation utilities
│   │   ├── light.h                    # Light source structures
│   │   ├── cuda_memory.h              # CUDA memory management
│   │   ├── cuda_utils.h               # CUDA utility functions
│   │   ├── camera_setup.h             # Camera configuration
│   │   ├── constants.h                # Global constants
│   │   ├── strong_types.h             # Type safety utilities
│   │   └── stb_image_write.h          # STB image writing library
│   └── cuda/                          # CUDA source files
│       ├── ray_trace_cuda.cu          # CUDA kernel implementation
│       ├── ray_trace_cuda.h           # CUDA kernel header
│       └── ray_trace_cuda_helper.cpp  # Helper functions for CUDA
├── GraphicsRendererUI/                # .NET WPF UI application (if exists)
│   ├── MainWindow.xaml                # Main window layout
│   ├── ViewModels/                    # MVVM view models
│   ├── Models/                        # Data models
│   └── RendererAPI.cs                 # P/Invoke bindings to native library
├── build-ui/                          # CMake build output directory
│   └── libGraphicsRendererAPI.so      # Shared library (Linux) or GraphicsRendererAPI.dll (Windows)
├── CMakeLists.txt                     # CMake build configuration
├── BUILD_AND_RUN.md                   # Build and run instructions
├── CUDA_README.md                     # CUDA implementation details
├── UI_README.md                       # UI application documentation
├── build-ui.sh                        # Build script
├── run-ui.sh                          # Run script
└── README.md                          # This file
```

### Key Source Directories

1. **`src/cpp/`** - Contains all C++ source and header files
   - Core rendering logic
   - API for .NET interop
   - Utilities (file I/O, transformations, etc.)
   - Header-only libraries

2. **`src/cuda/`** - Contains CUDA-specific source files
   - CUDA kernels for parallel ray tracing
   - CUDA helper functions
   - Device-side ray tracing implementation

### Important Files

#### Core Rendering Files
- **`src/cpp/main.cpp`** - Command-line application entry point
- **`src/cuda/ray_trace_cuda.cu`** - Main CUDA kernel for parallel ray tracing
- **`src/cuda/ray_trace_cuda_helper.cpp`** - CPU-side helpers for CUDA operations

#### API and Interop
- **`src/cpp/renderer_api.cpp/.h`** - C API wrapper for .NET P/Invoke
- **`GraphicsRendererUI/RendererAPI.cs`** - .NET bindings to native library

#### Utilities
- **`src/cpp/obj_loader.cpp/.h`** - Load 3D models from OBJ files
- **`src/cpp/jpg_writer.cpp`** - Write rendered images as JPEG
- **`src/cpp/ppm_to_jpg.cpp`** - Standalone utility to convert PPM to JPEG

### Build Targets

The CMakeLists.txt defines three build targets:

1. **`GraphicsRenderer`** - Command-line executable
   - Entry point: `src/cpp/main.cpp`
   - Outputs PPM images to stdout

2. **`GraphicsRendererAPI`** - Shared library for .NET interop
   - Entry point: `src/cpp/renderer_api.cpp`
   - Outputs: `libGraphicsRendererAPI.so` (Linux) or `GraphicsRendererAPI.dll` (Windows)

3. **`ppm_to_jpg`** - PPM to JPEG converter utility
   - Entry point: `src/cpp/ppm_to_jpg.cpp`

## Getting Started

### Prerequisites

- CUDA Toolkit (11.5 or later recommended)
- CMake (3.18 or later)
- C++ compiler with C++17 support
- .NET 8.0 SDK (for UI application)
- CUDA-capable GPU (compute capability 5.0 or higher)

### Building

See [BUILD_AND_RUN.md](BUILD_AND_RUN.md) for detailed build instructions.

Quick build:
```bash
./build-ui.sh
```

### Running

See [BUILD_AND_RUN.md](BUILD_AND_RUN.md) for detailed run instructions.

Quick run:
```bash
./run-ui.sh
```

## Documentation

- **[BUILD_AND_RUN.md](BUILD_AND_RUN.md)** - Complete build and run instructions
- **[CUDA_README.md](CUDA_README.md)** - CUDA implementation details and architecture
- **[UI_README.md](UI_README.md)** - UI application features and usage

## Architecture

### CUDA Acceleration

The rendering engine uses CUDA to parallelize ray tracing calculations:

1. Scene data is copied to GPU memory
2. A 2D grid of CUDA threads is launched (one thread per pixel)
3. Each thread independently traces rays and computes pixel colors
4. The rendered image is copied back to host memory

See [CUDA_README.md](CUDA_README.md) for more details.

### .NET UI Integration

The .NET WPF application provides a user-friendly interface:

- Add and configure 3D objects (pyramids, boxes, custom OBJ models)
- Adjust transforms (position, scale, rotation)
- Configure render settings
- Generate rendered images

See [UI_README.md](UI_README.md) for more details.

## Current Status

**Note**: The source files referenced in this documentation and the CMakeLists.txt are expected to be located in `src/cpp/` and `src/cuda/` directories. 

**If these directories do not exist in your clone:**
- The source files may need to be added to the repository
- Check with the repository maintainer for the location of the source code
- The build artifacts in `build-ui/` indicate that source files existed at build time
- You may need to restore the source files from a backup or alternative branch

## Contributing

When adding source files to this project, please follow the directory structure outlined above to maintain consistency with the build configuration.

## License

[License information to be added]
