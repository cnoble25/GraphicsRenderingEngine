#!/bin/bash

# Run script for Graphics Rendering Engine
# Starts the .NET UI application

set -e  # Exit on error

echo "═══════════════════════════════════════"
echo "  Starting Graphics Rendering Engine"
echo "═══════════════════════════════════════"
echo ""

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if library exists
if [ ! -f "build-ui/libGraphicsRendererAPI.so" ]; then
    echo "⚠️  Warning: Library not found. Building first..."
    ./build.sh
fi

# Check if UI is built
if [ ! -f "ui/bin/Release/net8.0/GraphicsRendererUI.dll" ]; then
    echo "⚠️  Warning: UI not built. Building first..."
    ./build.sh
fi

# Stop any existing instance
echo "🛑 Stopping any existing instances..."
pkill -f "dotnet GraphicsRendererUI.dll" 2>/dev/null || true
sleep 1

# Set up environment
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
export LD_LIBRARY_PATH="$SCRIPT_DIR/build-ui:$LD_LIBRARY_PATH"

# Change to UI directory
cd ui/bin/Release/net8.0

# Run the application
echo "🚀 Starting application..."
echo ""
dotnet GraphicsRendererUI.dll "$@"
