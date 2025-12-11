# Build Requirements

This document lists all the dependencies and tools needed to build and run the Graphics Rendering Engine.

## Required Tools

### 1. C++ Build System
- **CMake** (version 3.22 or higher)
  - Install: `sudo apt-get install cmake` (Ubuntu/Debian)
  - Or download from: https://cmake.org/download/
- **Make** (usually pre-installed)
  - Install: `sudo apt-get install build-essential` (Ubuntu/Debian)

### 2. C++ Compiler
- **GCC** or **Clang** with C++17 support
  - GCC 7+ or Clang 5+ recommended
  - Install: `sudo apt-get install g++` (Ubuntu/Debian)

### 3. CUDA Toolkit
- **CUDA Toolkit** (version 11.0 or higher)
  - Required for ray tracing functionality
  - Download from: https://developer.nvidia.com/cuda-downloads
  - Install: Follow NVIDIA's installation guide for your OS
  - Verify: `nvcc --version`

### 4. .NET SDK
- **.NET SDK 8.0** (or higher)
  - Download from: https://dotnet.microsoft.com/download
  - Install: Follow Microsoft's installation guide
  - Verify: `dotnet --version`

## Quick Setup (Ubuntu/Debian)

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y cmake build-essential

# Install CUDA (follow NVIDIA's guide for your system)
# https://developer.nvidia.com/cuda-downloads

# Install .NET SDK 8.0
wget https://dotnet.microsoft.com/download/dotnet/scripts/v1/dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0
export PATH="$HOME/.dotnet:$PATH"

# Verify installations
cmake --version
make --version
dotnet --version
nvcc --version
```

## Build and Run

Once all dependencies are installed:

```bash
# Build the project
./build.sh

# Run the application
./run.sh
```

## Project Structure

- `build-ui/` - CMake build directory for C++ library
- `ui/` - .NET Avalonia UI application
- `src/cpp/` - C++ source code
- `src/cuda/` - CUDA ray tracing code

## Troubleshooting

### CUDA not found
- Ensure CUDA is installed and `nvcc` is in your PATH
- Check: `which nvcc`
- Set CUDA_PATH if needed: `export CUDA_PATH=/usr/local/cuda`

### .NET not found
- Ensure .NET SDK is installed: `dotnet --version`
- Add to PATH: `export PATH="$HOME/.dotnet:$PATH"`
- Or install system-wide: `sudo apt-get install dotnet-sdk-8.0`

### Library not found at runtime
- The `run.sh` script sets `LD_LIBRARY_PATH` automatically
- If running manually, set: `export LD_LIBRARY_PATH=/path/to/build-ui:$LD_LIBRARY_PATH`

## Optional Dependencies

- **Git** - For version control
- **IDE** (VS Code, Visual Studio, etc.) - For development
- **GPU** - NVIDIA GPU with CUDA support (required for ray tracing)
