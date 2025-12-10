#!/bin/bash

# Script to run GraphicsRendererUI application
# This ensures the native library can be found and display is configured

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set DISPLAY if not already set (for WSL2)
if [ -z "$DISPLAY" ]; then
    # Try to detect WSL2 and set DISPLAY to Windows host
    if [ -f /proc/version ] && grep -q Microsoft /proc/version; then
        export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
        echo "WSL2 detected. Setting DISPLAY to: $DISPLAY"
    else
        export DISPLAY=:0
        echo "Setting DISPLAY to: $DISPLAY"
    fi
fi

# Ensure the native library is in the output directory
# The library should be copied during build, but verify it exists
BUILD_CONFIG="${1:-Release}"
LIB_PATH="bin/${BUILD_CONFIG}/net8.0/libGraphicsRendererAPI.so"

if [ ! -f "$LIB_PATH" ]; then
    echo "Warning: Native library not found at $LIB_PATH"
    echo "Attempting to build the project..."
    dotnet build -c "$BUILD_CONFIG"
fi

# Add library directory to LD_LIBRARY_PATH so .NET can find it
export LD_LIBRARY_PATH="$(pwd)/bin/${BUILD_CONFIG}/net8.0:${LD_LIBRARY_PATH}"

echo "Running GraphicsRendererUI..."
echo "DISPLAY=$DISPLAY"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# Run the application
dotnet run --configuration "$BUILD_CONFIG" "$@"
