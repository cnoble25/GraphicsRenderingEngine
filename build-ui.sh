#!/bin/bash

# Build script for Graphics Renderer UI
# This builds both the native library and the .NET UI application

set -e

echo "Building Graphics Renderer UI..."

# Create build directory
mkdir -p build-ui
cd build-ui

# Configure CMake
echo "Configuring CMake..."
cmake ..

# Build the shared library
echo "Building native library..."
make GraphicsRendererAPI

# Return to project root
cd ..

# Build .NET application
echo "Building .NET UI application..."
cd ui

# Set DOTNET_ROOT to use the proper .NET installation
# The system /usr/bin/dotnet points to an incomplete installation
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"

# Use the dotnet from ~/.dotnet directly to avoid the broken symlink
DOTNET_CMD="$HOME/.dotnet/dotnet"
if [ ! -f "$DOTNET_CMD" ]; then
    echo "Error: .NET SDK not found at $DOTNET_CMD"
    exit 1
fi

"$DOTNET_CMD" restore
"$DOTNET_CMD" build -c Release

echo "Build complete!"
echo ""
echo "To run the application:"
echo "  cd ui"
echo "  dotnet run"
echo ""
echo "Make sure libGraphicsRendererAPI.so is in the same directory or set LD_LIBRARY_PATH"
