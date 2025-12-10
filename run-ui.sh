#!/bin/bash

# Run script for Graphics Renderer UI
# This script sets up the environment and runs the application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build-ui"
UI_DIR="$SCRIPT_DIR/ui"
BIN_DIR="$UI_DIR/bin/Release/net8.0"
LIB_NAME="libGraphicsRendererAPI.so"

# Set DOTNET_ROOT
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"

# Check if the native library exists
if [ ! -f "$BUILD_DIR/$LIB_NAME" ]; then
    echo "Error: Native library not found at $BUILD_DIR/$LIB_NAME"
    echo "Please run ./build-ui.sh first to build the project."
    exit 1
fi

# Check if the application is built
if [ ! -f "$BIN_DIR/GraphicsRendererUI.dll" ]; then
    echo "Error: Application not built. Building now..."
    cd "$SCRIPT_DIR"
    ./build-ui.sh
fi

# Set LD_LIBRARY_PATH so the application can find the native library
export LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH"

# Copy the library to the bin directory for easier access (optional)
if [ ! -f "$BIN_DIR/$LIB_NAME" ]; then
    echo "Copying native library to bin directory..."
    cp "$BUILD_DIR/$LIB_NAME" "$BIN_DIR/"
fi

# Run the application
echo "Running Graphics Renderer UI..."
echo "Native library location: $BUILD_DIR/$LIB_NAME"
echo ""
cd "$BIN_DIR"
dotnet GraphicsRendererUI.dll
