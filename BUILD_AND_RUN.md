# Building and Running Graphics Renderer UI

## Prerequisites

- .NET SDK 8.0 (already configured)
- CMake
- CUDA toolkit (for native library compilation)
- Make

## Building the Project

### Option 1: Using the Build Script (Recommended)

From the project root directory (`/root/projects/GraphicsRenderingEngine`):

```bash
./build-ui.sh
```

This script will:
1. Build the native library (`libGraphicsRendererAPI.so`) using CMake
2. Build the .NET UI application
3. Place outputs in `build-ui/` and `GraphicsRendererUI/bin/Release/net8.0/`

### Option 2: Manual Build

#### Build Native Library:
```bash
mkdir -p build-ui
cd build-ui
cmake ..
make GraphicsRendererAPI
cd ..
```

#### Build .NET Application:
```bash
cd GraphicsRendererUI
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
dotnet restore
dotnet build -c Release
```

## Running the Application

### Option 1: Using the Run Script (Recommended)

From the project root directory:

```bash
./run-ui.sh
```

This script automatically:
- Sets up the environment variables
- Ensures the native library is accessible
- Runs the application

### Option 2: Manual Run

#### Step 1: Set up environment variables
```bash
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
export LD_LIBRARY_PATH="/root/projects/GraphicsRenderingEngine/build-ui:$LD_LIBRARY_PATH"
```

#### Step 2: Navigate to the build directory
```bash
cd GraphicsRendererUI/bin/Release/net8.0
```

#### Step 3: Run the application
```bash
dotnet GraphicsRendererUI.dll
```

Or from the GraphicsRendererUI directory:
```bash
cd GraphicsRendererUI
dotnet run -c Release
```

### Option 3: Copy Library and Run

You can also copy the native library to the bin directory:

```bash
cp build-ui/libGraphicsRendererAPI.so GraphicsRendererUI/bin/Release/net8.0/
cd GraphicsRendererUI/bin/Release/net8.0
dotnet GraphicsRendererUI.dll
```

## Quick Start

```bash
# Build everything
./build-ui.sh

# Run the application
./run-ui.sh
```

## Troubleshooting

### Error: "libGraphicsRendererAPI.so: cannot open shared object file"

Make sure `LD_LIBRARY_PATH` includes the directory containing the library:
```bash
export LD_LIBRARY_PATH="/root/projects/GraphicsRenderingEngine/build-ui:$LD_LIBRARY_PATH"
```

### Error: "dotnet: command not found"

Make sure the environment variables are set:
```bash
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
```

Or source your `.bashrc`:
```bash
source ~/.bashrc
```

## Project Structure

```
GraphicsRenderingEngine/
├── build-ui/                          # CMake build directory
│   └── libGraphicsRendererAPI.so      # Native library (built)
├── GraphicsRendererUI/                # .NET UI application
│   ├── bin/Release/net8.0/           # Build output
│   │   └── GraphicsRendererUI.dll    # Main application
│   └── ...
├── build-ui.sh                        # Build script
└── run-ui.sh                          # Run script
```
