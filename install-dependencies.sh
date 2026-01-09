#!/bin/bash
# Installation script for Graphics Rendering Engine dependencies
# Run with: sudo ./install-dependencies.sh

set -e  # Exit on error

echo "========================================="
echo "Installing Graphics Rendering Engine Dependencies"
echo "========================================="

# Detect Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_VERSION=$VERSION_ID
else
    echo "Warning: Could not detect Ubuntu version, assuming 24.04"
    UBUNTU_VERSION="24.04"
fi

# Update package list
echo ""
echo "Step 1: Updating package list..."
apt-get update

# Install CMake and build tools
echo ""
echo "Step 2: Installing CMake and build-essential (make, g++)..."
apt-get install -y cmake build-essential

# Install CUDA Toolkit
echo ""
echo "Step 3: Installing CUDA Toolkit..."
echo "Note: This may take a while and requires significant disk space..."
echo "For WSL2, ensure NVIDIA drivers are installed on Windows host first."
apt-get install -y nvidia-cuda-toolkit

# Install .NET SDK 8.0 (skip if already installed)
echo ""
echo "Step 4: Checking .NET SDK 8.0 installation..."
if command -v dotnet >/dev/null 2>&1; then
    DOTNET_VERSION=$(dotnet --version 2>/dev/null || echo "")
    if [ -n "$DOTNET_VERSION" ]; then
        echo ".NET SDK is already installed (version: $DOTNET_VERSION). Skipping..."
    else
        echo "Installing .NET SDK 8.0 via package manager..."
        # Add Microsoft package repository for .NET
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR"
        wget https://packages.microsoft.com/config/ubuntu/${UBUNTU_VERSION}/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
        dpkg -i packages-microsoft-prod.deb
        rm packages-microsoft-prod.deb
        cd -
        rmdir "$TEMP_DIR" 2>/dev/null || true
        apt-get update
        apt-get install -y dotnet-sdk-8.0
    fi
else
    echo "Installing .NET SDK 8.0 via package manager..."
    # Add Microsoft package repository for .NET
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    wget https://packages.microsoft.com/config/ubuntu/${UBUNTU_VERSION}/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
    dpkg -i packages-microsoft-prod.deb
    rm packages-microsoft-prod.deb
    cd -
    rmdir "$TEMP_DIR" 2>/dev/null || true
    apt-get update
    apt-get install -y dotnet-sdk-8.0
fi

# Verify installations
echo ""
echo "========================================="
echo "Verifying installations..."
echo "========================================="

echo ""
echo "CMake version:"
cmake --version

echo ""
echo "G++ version:"
g++ --version

echo ""
echo "CUDA version:"
nvcc --version

echo ""
echo ".NET version:"
dotnet --version

echo ""
echo "========================================="
echo "All dependencies installed successfully!"
echo "========================================="
echo ""
echo "You can now build the project with:"
echo "  ./build-ui.sh"
echo ""
echo "And run it with:"
echo "  ./run-ui.sh"
echo ""
