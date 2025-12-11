# Windows Build Instructions

This guide will help you build and run the Graphics Renderer UI application on Windows.

## Prerequisites

1. **Visual Studio 2022** (or later) with C++ development tools
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Make sure to install "Desktop development with C++" workload
   - Include "Windows 10/11 SDK"

2. **CMake** (3.18 or later)
   - Download from: https://cmake.org/download/
   - Add CMake to your PATH during installation

3. **CUDA Toolkit** (optional, for ray tracing)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Required for ray tracing mode
   - Rasterization mode will work without CUDA

4. **.NET 8.0 SDK**
   - Download from: https://dotnet.microsoft.com/download/dotnet/8.0
   - Make sure to install the SDK (not just runtime)

## Building the Application

### Option 1: Using PowerShell (Recommended)

1. Open PowerShell in the project root directory
2. Run:
   ```powershell
   .\build-ui.ps1
   ```

### Option 2: Using Command Prompt

1. Open Command Prompt in the project root directory
2. Run:
   ```cmd
   build-ui.bat
   ```

### Option 3: Manual Build

1. **Build the native library:**
   ```cmd
   mkdir build-ui
   cd build-ui
   cmake .. -G "Visual Studio 17 2022" -A x64
   cmake --build . --config Release --target GraphicsRendererAPI
   cd ..
   ```

2. **Build the .NET application:**
   ```cmd
   cd GraphicsRendererUI
   dotnet restore
   dotnet build -c Release
   ```

3. **Copy the DLL:**
   ```cmd
   copy ..\build-ui\Release\GraphicsRendererAPI.dll bin\Release\net8.0\GraphicsRendererAPI.dll
   ```

## Running the Application

1. Navigate to the GraphicsRendererUI directory:
   ```cmd
   cd GraphicsRendererUI
   ```

2. Run the application:
   ```cmd
   dotnet run -c Release
   ```

   Or run the executable directly:
   ```cmd
   bin\Release\net8.0\GraphicsRendererUI.exe
   ```

## Troubleshooting

### CMake can't find Visual Studio
- Make sure Visual Studio 2022 is installed with C++ tools
- Try specifying a different generator:
  ```cmd
  cmake .. -G "Visual Studio 16 2019" -A x64
  ```
  Or use MinGW:
  ```cmd
  cmake .. -G "MinGW Makefiles"
  ```

### CUDA not found
- Ray tracing requires CUDA. If CUDA is not installed, only rasterization mode will work.
- Make sure CUDA Toolkit is installed and `nvcc` is in your PATH
- Set `CUDA_PATH` environment variable if needed

### DLL not found at runtime
- Make sure `GraphicsRendererAPI.dll` is in the same directory as `GraphicsRendererUI.exe`
- Or copy it to `bin\Release\net8.0\` directory

### .NET SDK not found
- Make sure .NET 8.0 SDK is installed (not just runtime)
- Verify installation: `dotnet --version`
- Add .NET to your PATH if needed

## File Structure

After building, your directory structure should look like:

```
GraphicsRendererUI/
├── bin/
│   └── Release/
│       └── net8.0/
│           ├── GraphicsRendererUI.exe
│           ├── GraphicsRendererAPI.dll  ← Must be here
│           └── ... (other DLLs)
└── ...
```

## Notes

- The application will automatically detect the platform and load the correct DLL
- On Windows, it looks for `GraphicsRendererAPI.dll`
- On Linux, it looks for `libGraphicsRendererAPI.so`
- On macOS, it looks for `libGraphicsRendererAPI.dylib`
