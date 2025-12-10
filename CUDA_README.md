# CUDA Ray Tracing Implementation

This project has been converted to use CUDA for parallel ray tracing calculations. The nested for loops in the main function that perform ray tracing are now executed in parallel on the GPU.

## Features

- **Parallel Ray Tracing**: All pixel calculations are performed in parallel on the GPU
- **Multiple Objects Support**: You can add multiple 3D models/objects to the scene
- **CUDA Acceleration**: Significant performance improvement for large images

## Building

### Prerequisites

- CUDA Toolkit (version 11.0 or later recommended)
- CMake (version 3.18 or later)
- A CUDA-capable GPU

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

If CMake doesn't detect CUDA automatically, you may need to specify the CUDA path:

```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

## Usage

### Adding Objects to the Scene

In `main.cpp`, you can add multiple models to the scene:

```cpp
std::vector<model> scene_objects;

// Add a pyramid
scene_objects.push_back(pyamid());

// Add a box
scene_objects.push_back(box());

// Add custom models
scene_objects.push_back(model({vertex1, vertex2, ...}, transforms(...)));
```

### Running

```bash
./GraphicsRenderer > output.ppm
```

The program outputs a PPM image file that can be viewed with any image viewer that supports PPM format.

## Architecture

### Files

- `ray_trace_cuda.cu`: CUDA kernel implementation with device-side ray tracing functions
- `ray_trace_cuda.h`: Header file with CUDA-compatible data structures
- `ray_trace_cuda_helper.cpp`: Helper functions to prepare models for CUDA
- `main.cpp`: Main program that sets up the scene and launches CUDA kernels

### How It Works

1. **Scene Setup**: Models are prepared and converted to CUDA-compatible format
2. **Memory Allocation**: GPU memory is allocated for triangles, model metadata, and output image
3. **Data Transfer**: Scene data is copied to GPU memory
4. **Kernel Launch**: CUDA kernel is launched with a 2D grid of threads (one per pixel)
5. **Parallel Execution**: Each thread computes the color for one pixel independently
6. **Result Retrieval**: The rendered image is copied back to host memory
7. **Output**: The image is written to stdout in PPM format

### Performance

The CUDA implementation parallelizes the two nested for loops that iterate over all pixels. Each pixel's ray tracing calculation runs in parallel on the GPU, providing significant speedup for large images.

## Customization

### Adjusting CUDA Architecture

If you have a specific GPU, you can optimize the build by setting the CUDA architecture in `CMakeLists.txt`:

```cmake
set_property(TARGET GraphicsRenderer PROPERTY CUDA_ARCHITECTURES "75")  # For RTX 20-series
```

### Changing Block Size

The kernel uses a 16x16 thread block by default. You can modify this in `ray_trace_cuda.cu`:

```cpp
dim3 blockSize(16, 16);  // Change to (32, 32) or other values
```

## Troubleshooting

### CUDA Not Found

If CMake can't find CUDA:
1. Ensure CUDA Toolkit is installed
2. Add CUDA to your PATH: `export PATH=/usr/local/cuda/bin:$PATH`
3. Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`

### Kernel Launch Failed

- Ensure you have a CUDA-capable GPU
- Check GPU compute capability matches the architectures specified in CMakeLists.txt
- Verify CUDA drivers are installed: `nvidia-smi`

### Out of Memory

If you get CUDA out-of-memory errors:
- Reduce image resolution
- Reduce the number of triangles/models in the scene
- Use a GPU with more memory
