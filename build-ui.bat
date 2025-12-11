@echo off
REM Build script for Graphics Renderer UI (Windows)
REM This builds both the native library and the .NET UI application

echo Building Graphics Renderer UI for Windows...

REM Check if CMake is available
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: CMake is not installed or not in PATH
    echo Please install CMake from https://cmake.org/download/
    exit /b 1
)

REM Check if CUDA is available
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Warning: CUDA compiler (nvcc) not found. Ray tracing will not work.
    echo Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
)

REM Create build directory
if not exist build-ui mkdir build-ui
cd build-ui

REM Configure CMake for Windows
echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64
if %ERRORLEVEL% NEQ 0 (
    echo Trying with MinGW Makefiles...
    cmake .. -G "MinGW Makefiles"
    if %ERRORLEVEL% NEQ 0 (
        echo Error: CMake configuration failed
        echo Please ensure you have a compatible generator installed
        exit /b 1
    )
)

REM Build the shared library
echo Building native library...
cmake --build . --config Release --target GraphicsRendererAPI
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build native library
    exit /b 1
)

REM Return to project root
cd ..

REM Build .NET application
echo Building .NET UI application...
cd ui

REM Check if dotnet is available
where dotnet >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: .NET SDK is not installed or not in PATH
    echo Please install .NET SDK from https://dotnet.microsoft.com/download
    exit /b 1
)

dotnet restore
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to restore .NET packages
    exit /b 1
)

dotnet build -c Release
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to build .NET application
    exit /b 1
)

REM Copy DLL to output directory
if exist "..\build-ui\Release\GraphicsRendererAPI.dll" (
    copy /Y "..\build-ui\Release\GraphicsRendererAPI.dll" "bin\Release\net8.0\GraphicsRendererAPI.dll"
) else if exist "..\build-ui\GraphicsRendererAPI.dll" (
    copy /Y "..\build-ui\GraphicsRendererAPI.dll" "bin\Release\net8.0\GraphicsRendererAPI.dll"
)

echo.
echo Build complete!
echo.
echo To run the application:
echo   cd ui
echo   dotnet run -c Release
echo.
echo Make sure GraphicsRendererAPI.dll is in the same directory as the executable
