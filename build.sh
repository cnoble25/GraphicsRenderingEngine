#!/bin/bash

# Build script for Graphics Rendering Engine
# Builds both C++ library and .NET UI

set -e  # Exit on error

echo "═══════════════════════════════════════"
echo "  Building Graphics Rendering Engine"
echo "═══════════════════════════════════════"
echo ""

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build C++ library
echo "📦 Building C++ library..."
cd build-ui
if ! make GraphicsRendererAPI 2>&1 | tee /tmp/build_cpp.log | tail -5; then
    echo "❌ C++ library build failed!"
    echo "Check /tmp/build_cpp.log for details"
    exit 1
fi
cd ..

# Copy library to UI directory
echo ""
echo "📋 Copying library to UI directory..."
cp build-ui/libGraphicsRendererAPI.so ui/bin/Release/net8.0/ || {
    echo "⚠️  Warning: Could not copy library (might not exist yet)"
}

# Build .NET UI
echo ""
echo "📦 Building .NET UI..."
cd ui
if ! dotnet build -c Release 2>&1 | tee /tmp/build_ui.log | tail -5; then
    echo "❌ .NET UI build failed!"
    echo "Check /tmp/build_ui.log for details"
    exit 1
fi
cd ..

echo ""
echo "═══════════════════════════════════════"
echo "  ✅ Build Complete!"
echo "═══════════════════════════════════════"
echo ""
echo "To run the application, use:"
echo "  ./run.sh"
echo ""
