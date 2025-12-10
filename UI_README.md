# Graphics Renderer UI

A .NET WPF application for managing and rendering 3D scenes with the CUDA-based Graphics Rendering Engine.

## Features

- **Add Objects**: Choose from built-in shapes (Pyramid, Box) or import custom OBJ files
- **Transform Controls**: Adjust position, scale, and rotation (roll, pitch, yaw) for each object
- **Scene Management**: Add, remove, and modify objects in your scene
- **Render Settings**: Configure image resolution, luminosity, and output path
- **Real-time Updates**: Transform changes are applied when rendering

## Building

### Prerequisites

- CMake 3.18 or higher
- CUDA toolkit
- C++ compiler with C++17 support
- .NET 8.0 SDK
- Linux/WSL2 (for Linux build) or Windows (for Windows build)

### Building the Native Library

1. Build the shared library (`libGraphicsRendererAPI.so` on Linux, `GraphicsRendererAPI.dll` on Windows):

```bash
mkdir -p build-ui
cd build-ui
cmake ..
make GraphicsRendererAPI
```

The shared library will be created in the `build-ui` directory.

### Building the .NET UI Application

1. Navigate to the UI project directory:

```bash
cd GraphicsRendererUI
```

2. Restore dependencies and build:

```bash
dotnet restore
dotnet build
```

3. Run the application:

```bash
dotnet run
```

Or build a self-contained executable:

```bash
dotnet publish -c Release -r linux-x64 --self-contained true
```

## Usage

1. **Add Objects**:
   - Click "Add Pyramid" or "Add Box" to add built-in shapes
   - Click "Import OBJ File" to load a custom 3D model

2. **Modify Object Properties**:
   - Select an object from the list on the left
   - Adjust Position (X, Y, Z), Scale (X, Y, Z), and Rotation (Roll, Pitch, Yaw) in the properties panel

3. **Render Scene**:
   - Configure render settings (width, height, luminosity, output path)
   - Click "Render Scene" to generate the PPM image

4. **Remove Objects**:
   - Select an object and click "Remove Selected"

## File Structure

```
GraphicsRenderingEngine/
├── renderer_api.h/cpp          # C API wrapper for .NET interop
├── obj_loader.h/cpp              # OBJ file loader
├── GraphicsRendererUI/           # .NET WPF application
│   ├── MainWindow.xaml          # Main UI layout
│   ├── ViewModels/               # MVVM view models
│   ├── Models/                   # Data models
│   └── RendererAPI.cs            # P/Invoke bindings
└── CMakeLists.txt                # Build configuration
```

## Notes

- **Windows Required**: WPF applications require Windows. While you can develop on WSL2, you'll need to run the application on a Windows machine or use Windows Subsystem for Linux with X11 forwarding (not recommended for WPF)
- The shared library (`libGraphicsRendererAPI.so` on Linux or `GraphicsRendererAPI.dll` on Windows) must be in the same directory as the executable or in a system library path
- OBJ files should use standard OBJ format with vertex positions (`v`) and faces (`f`)
- The renderer outputs PPM format images
- CUDA-capable GPU is required for rendering

## Troubleshooting

- **Library not found**: Ensure the shared library is in the same directory as the executable or set `LD_LIBRARY_PATH` (Linux) or add to PATH (Windows)
- **CUDA errors**: Verify CUDA is installed and a CUDA-capable GPU is available
- **OBJ file loading fails**: Check that the OBJ file path is correct and the file format is valid
