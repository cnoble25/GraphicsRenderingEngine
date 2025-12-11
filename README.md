# Graphics Rendering Engine

A high-performance graphics rendering engine that supports both ray tracing (CUDA-accelerated) and rasterization rendering modes. The project consists of a C++/CUDA backend library and a .NET/Avalonia UI frontend.

## Project Structure

The project is organized into three main sections:

### 📁 C++ Libraries (`src/cpp/`)

Contains all C++ library code, headers, and the main executable entry point.

**Source Files:**
- `main.cpp` - Main entry point for the standalone executable
- `obj_loader.cpp` / `obj_loader.h` - OBJ file loading functionality
- `renderer_api.cpp` / `renderer_api.h` - C API for .NET interop (shared library interface)

**Header Files:**
- `color.h` - Color representation and utilities
- `vec3.h` - 3D vector mathematics
- `ray.h` - Ray representation for ray tracing
- `vertex.h` - Vertex data structures
- `model.h` - 3D model representation and transformations
- `transform.h` - Transformation matrices
- `rotation.h` - Rotation utilities
- `rasterization.h` - Rasterization rendering implementation

**Purpose:** Core rendering logic, scene management, and API interfaces.

### 📁 CUDA (`src/cuda/`)

Contains all CUDA-specific code for GPU-accelerated ray tracing.

**Files:**
- `ray_trace_cuda.cu` - CUDA kernel implementation for parallel ray tracing
- `ray_trace_cuda.h` - CUDA data structures and function declarations
- `ray_trace_cuda_helper.cpp` - Helper functions for preparing data for CUDA kernels

**Purpose:** GPU-accelerated ray tracing implementation using CUDA.

### 📁 .NET UI (`ui/`)

Contains the .NET/Avalonia-based user interface application.

**Main Files:**
- `App.xaml` / `App.xaml.cs` - Application entry point and configuration
- `MainWindow.xaml` / `MainWindow.xaml.cs` - Main window UI and code-behind
- `GraphicsRendererUI.csproj` - .NET project file

**Models:**
- `Models/SceneObject.cs` - Scene object data model

**ViewModels:**
- `ViewModels/MainViewModel.cs` - Main window view model (MVVM pattern)

**Utils:**
- `Utils/PpmDecoder.cs` - PPM image format decoder

**Supporting Files:**
- `RendererAPI.cs` - P/Invoke wrapper for native library
- `ObservableObject.cs` - Base class for MVVM observable objects
- `RelayCommand.cs` - Command implementation for MVVM

**Purpose:** Cross-platform desktop UI for scene creation, editing, and rendering.

## Build System

### CMake Configuration

The project uses CMake for building the C++/CUDA components. The main `CMakeLists.txt` defines:

- **GraphicsRenderer** - Standalone executable
- **GraphicsRendererAPI** - Shared library for .NET interop

Both targets include:
- C++ standard: C++17
- CUDA standard: CUDA 17
- CUDA architectures: 50, 60, 70, 75, 80, 86

### Build Scripts

**Linux/macOS:**
- `build-ui.sh` - Builds both native library and .NET UI
- `run-ui.sh` - Runs the UI application with proper library paths

**Windows:**
- `build-ui.bat` - Batch script for building on Windows
- `build-ui.ps1` - PowerShell script for building on Windows

## Building the Project

### Prerequisites

- CMake 3.18 or higher
- C++17 compatible compiler (GCC, Clang, or MSVC)
- CUDA Toolkit (for ray tracing support)
- .NET SDK 8.0 or higher (for UI)

### Build Steps

**Linux/macOS:**
```bash
./build-ui.sh
```

**Windows:**
```cmd
build-ui.bat
```
or
```powershell
.\build-ui.ps1
```

**Manual Build:**
```bash
# Create build directory
mkdir -p build-ui
cd build-ui

# Configure CMake
cmake ..

# Build native library
make GraphicsRendererAPI  # or cmake --build . --target GraphicsRendererAPI

# Build .NET UI (from project root)
cd ../ui
dotnet restore
dotnet build -c Release
```

## Running the Application

### Standalone Executable

```bash
cd build-ui
./GraphicsRenderer
```

This generates `output.ppm` in the current directory.

### UI Application

**Linux/macOS:**
```bash
./run-ui.sh
```

**Or manually:**
```bash
cd ui
export LD_LIBRARY_PATH=../build-ui:$LD_LIBRARY_PATH
dotnet run
```

**Windows:**
```cmd
cd ui
dotnet run -c Release
```

## Output

The renderer outputs images in PPM (Portable Pixmap) format. All rendered images are stored in the `renders/` directory at the project root:

- **Standalone executable**: Creates `renders/output.ppm` (automatically creates the directory if it doesn't exist)
- **UI application**: Defaults to `renders/output.ppm`, but allows you to specify a custom output path

The `renders/` directory is automatically created if it doesn't exist when rendering.

## Rendering Modes

1. **Ray Tracing (CUDA)** - GPU-accelerated ray tracing for high-quality rendering
2. **Rasterization** - CPU-based rasterization for compatibility

## Dependencies

### C++/CUDA
- CUDA Runtime
- Standard C++ Library

### .NET
- Avalonia UI Framework 11.0.0
- .NET 8.0 Runtime

## File Organization Summary

```
GraphicsRenderingEngine/
├── src/
│   ├── cpp/              # C++ library code
│   │   ├── *.cpp         # Source files
│   │   └── *.h           # Header files
│   └── cuda/             # CUDA code
│       ├── *.cu          # CUDA kernel files
│       ├── *.h           # CUDA headers
│       └── *.cpp         # CUDA helper code
├── ui/                   # .NET UI application
│   ├── *.cs              # C# source files
│   ├── *.xaml            # UI markup files
│   ├── Models/           # Data models
│   ├── ViewModels/       # MVVM view models
│   └── Utils/            # Utility classes
├── renders/              # Output directory for PPM images
│   └── *.ppm             # Rendered image files
├── build-ui/             # Build output directory
├── CMakeLists.txt        # CMake configuration
└── build scripts         # Platform-specific build scripts
```

## Additional Documentation

- `CUDA_README.md` - CUDA-specific documentation
- `UI_README.md` - UI application documentation
- `WINDOWS_BUILD.md` / `README_WINDOWS.md` - Windows-specific build instructions

## License

[Add your license information here]
