# Build script for Graphics Renderer UI (Windows PowerShell)
# This builds both the native library and the .NET UI application

Write-Host "Building Graphics Renderer UI for Windows..." -ForegroundColor Green

# Check if CMake is available
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "Error: CMake is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install CMake from https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

# Check if CUDA is available
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Write-Host "Warning: CUDA compiler (nvcc) not found. Ray tracing will not work." -ForegroundColor Yellow
    Write-Host "Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

# Create build directory
if (-not (Test-Path "build-ui")) {
    New-Item -ItemType Directory -Path "build-ui" | Out-Null
}
Set-Location "build-ui"

# Configure CMake for Windows
Write-Host "Configuring CMake..." -ForegroundColor Cyan
$cmakeGen = "Visual Studio 17 2022"
$cmakeArch = "-A x64"

cmake .. -G $cmakeGen $cmakeArch
if ($LASTEXITCODE -ne 0) {
    Write-Host "Trying with MinGW Makefiles..." -ForegroundColor Yellow
    cmake .. -G "MinGW Makefiles"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: CMake configuration failed" -ForegroundColor Red
        Write-Host "Please ensure you have a compatible generator installed" -ForegroundColor Yellow
        exit 1
    }
}

# Build the shared library
Write-Host "Building native library..." -ForegroundColor Cyan
cmake --build . --config Release --target GraphicsRendererAPI
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build native library" -ForegroundColor Red
    exit 1
}

# Return to project root
Set-Location ".."

# Build .NET application
Write-Host "Building .NET UI application..." -ForegroundColor Cyan
Set-Location "ui"

# Check if dotnet is available
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Host "Error: .NET SDK is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install .NET SDK from https://dotnet.microsoft.com/download" -ForegroundColor Yellow
    exit 1
}

dotnet restore
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to restore .NET packages" -ForegroundColor Red
    exit 1
}

dotnet build -c Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build .NET application" -ForegroundColor Red
    exit 1
}

# Copy DLL to output directory
$dllPath = "..\build-ui\Release\GraphicsRendererAPI.dll"
if (-not (Test-Path $dllPath)) {
    $dllPath = "..\build-ui\GraphicsRendererAPI.dll"
}

if (Test-Path $dllPath) {
    Copy-Item -Path $dllPath -Destination "bin\Release\net8.0\GraphicsRendererAPI.dll" -Force
    Write-Host "Copied GraphicsRendererAPI.dll to output directory" -ForegroundColor Green
} else {
    Write-Host "Warning: GraphicsRendererAPI.dll not found. Please copy it manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "  cd ui" -ForegroundColor White
Write-Host "  dotnet run -c Release" -ForegroundColor White
Write-Host ""
Write-Host "Make sure GraphicsRendererAPI.dll is in the same directory as the executable" -ForegroundColor Yellow
