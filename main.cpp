#include "color.h"
#include "vec3.h"
#include "ray.h"
#include <vector>
#include <cmath>
#include <iostream>
#include "vertex.h"
#include "model.h"
#include "transform.h"
#include "rotation.h"
#include <fstream>
#include "ray_trace_cuda.h"
#include <cuda_runtime.h>

int main() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        return 1;
    }

    // Define scene objects (models) - you can add more models here
    std::vector<model> scene_objects;
    
    // Add a pyramid to the scene
    scene_objects.push_back(pyamid());
    
    // You can add more objects like this:
    // scene_objects.push_back(box());
    // scene_objects.push_back(model({...}, transforms(...)));

    // Prepare models for CUDA
    std::vector<Vertex_cuda> all_triangles;
    std::vector<int> triangle_counts;
    std::vector<int> triangle_offsets;
    prepare_models_for_cuda(scene_objects, all_triangles, triangle_counts, triangle_offsets);

    if (all_triangles.empty()) {
        std::cerr << "No triangles in scene!" << std::endl;
        return 1;
    }

    // Image setup
    auto aspect_ratio = 16.0/9.0;
    int image_width = 800;
    int image_height = int(image_width/aspect_ratio);
    image_height = (image_height < 1) ? 1: image_height;

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(aspect_ratio));
    auto camera_center = point3(0,0,0);

    auto viewport_u = vec3(viewport_width,0,0);
    auto viewport_v = vec3(0,-viewport_height,0);

    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel100_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Convert camera parameters to CUDA format
    Vec3_cuda cuda_camera_center = vec3_to_cuda(camera_center);
    Vec3_cuda cuda_pixel100_loc = vec3_to_cuda(pixel100_loc);
    Vec3_cuda cuda_pixel_delta_u = vec3_to_cuda(pixel_delta_u);
    Vec3_cuda cuda_pixel_delta_v = vec3_to_cuda(pixel_delta_v);

    // Allocate device memory for triangles
    Vertex_cuda* d_models = nullptr;
    int* d_triangle_counts = nullptr;
    int* d_triangle_offsets = nullptr;
    Color_cuda* d_image = nullptr;

    size_t triangles_size = all_triangles.size() * sizeof(Vertex_cuda);
    size_t counts_size = triangle_counts.size() * sizeof(int);
    size_t offsets_size = triangle_offsets.size() * sizeof(int);
    size_t image_size = image_width * image_height * sizeof(Color_cuda);

    // Declare vector early to avoid crossing initialization with goto
    std::vector<Color_cuda> image(image_width * image_height);
    
    // Declare file stream early to avoid crossing initialization with goto
    std::ofstream outfile("output.ppm");

    cudaStatus = cudaMalloc((void**)&d_models, triangles_size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_models!" << std::endl;
        goto cleanup;
    }

    cudaStatus = cudaMalloc((void**)&d_triangle_counts, counts_size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_triangle_counts!" << std::endl;
        goto cleanup;
    }

    cudaStatus = cudaMalloc((void**)&d_triangle_offsets, offsets_size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_triangle_offsets!" << std::endl;
        goto cleanup;
    }

    cudaStatus = cudaMalloc((void**)&d_image, image_size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_image!" << std::endl;
        goto cleanup;
    }

    // Copy data to device
    cudaStatus = cudaMemcpy(d_models, all_triangles.data(), triangles_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_models!" << std::endl;
        goto cleanup;
    }

    cudaStatus = cudaMemcpy(d_triangle_counts, triangle_counts.data(), counts_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_triangle_counts!" << std::endl;
        goto cleanup;
    }

    cudaStatus = cudaMemcpy(d_triangle_offsets, triangle_offsets.data(), offsets_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for d_triangle_offsets!" << std::endl;
        goto cleanup;
    }

    // Launch CUDA kernel
    launch_render_kernel(
        d_image,
        image_width,
        image_height,
        cuda_camera_center,
        cuda_pixel100_loc,
        cuda_pixel_delta_u,
        cuda_pixel_delta_v,
        d_models,
        d_triangle_counts,
        d_triangle_offsets,
        scene_objects.size(),
        5.0  // luminosity
    );

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        goto cleanup;
    }

    // Wait for kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed!" << std::endl;
        goto cleanup;
    }

    // Copy result back to host
    cudaStatus = cudaMemcpy(image.data(), d_image, image_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for image!" << std::endl;
        goto cleanup;
    }

    // Output PPM image to file
    // Output in same order as original: from top row to bottom row
    // In CUDA kernel: j=0 corresponds to original j=image_height (top row)
    //                 j=image_height-1 corresponds to original j=1 (bottom row)
    // Original outputs: j from image_height down to 1, i from image_width down to 1
    // So we output: j from 0 to image_height-1, i from 0 to image_width-1
    // But need to reverse i to match original right-to-left order
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output.ppm for writing!" << std::endl;
        goto cleanup;
    }
    
    outfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = image_width - 1; i >= 0; i--) {
            int idx = j * image_width + i;
            color pixel_color = cuda_to_color(image[idx]);
            write_color(outfile, pixel_color);
        }
    }
    
    outfile.close();
    std::cout << "Image saved to output.ppm" << std::endl;

cleanup:
    // Free device memory
    if (d_models) cudaFree(d_models);
    if (d_triangle_counts) cudaFree(d_triangle_counts);
    if (d_triangle_offsets) cudaFree(d_triangle_offsets);
    if (d_image) cudaFree(d_image);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    return 0;
}