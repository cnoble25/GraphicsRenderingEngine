#ifndef RAY_TRACE_CUDA_H
#define RAY_TRACE_CUDA_H

#include "vec3.h"
#include "model.h"
#include "color.h"
#include <vector>

// CUDA-compatible structures (matching the .cu file)
struct Vec3_cuda {
    double x, y, z;
};

struct Color_cuda {
    double r, g, b;
};

struct Vertex_cuda {
    Vec3_cuda p0, p1, p2;
};

struct Light_cuda {
    Vec3_cuda position;
    Color_cuda color;
    double luminosity;
};

// Structure for custom pixel group assignment
// Each CUDA core processes one PixelGroup
struct PixelCoord_cuda {
    int x, y;  // Pixel coordinates
};

// Function to convert vec3 to Vec3_cuda
inline Vec3_cuda vec3_to_cuda(const vec3& v) {
    Vec3_cuda result;
    result.x = v.x();
    result.y = v.y();
    result.z = v.z();
    return result;
}

// Function to convert vertex to Vertex_cuda
inline Vertex_cuda vertex_to_cuda(const vertex& v) {
    Vertex_cuda result;
    result.p0 = vec3_to_cuda(v.first);
    result.p1 = vec3_to_cuda(v.second);
    result.p2 = vec3_to_cuda(v.third);
    return result;
}

// Function to convert Color_cuda to color
inline color cuda_to_color(const Color_cuda& c) {
    return color(c.r, c.g, c.b);
}

// CUDA memory management and kernel launch
extern "C" {
    // Standard block-based compression kernel
    void launch_render_kernel(
        Color_cuda* d_image,
        int image_width,
        int image_height,
        Vec3_cuda camera_center,
        Vec3_cuda pixel100_loc,
        Vec3_cuda pixel_delta_u,
        Vec3_cuda pixel_delta_v,
        const Vertex_cuda* d_models,
        const int* d_model_triangle_counts,
        const int* d_model_triangle_offsets,
        int num_models,
        const Light_cuda* d_lights,
        int num_lights,
        const double* d_light_absorptions,
        int max_bounces,
        int compression_level
    );
    
    // Custom compression algorithm kernel - assigns custom pixel groups to CUDA cores
    // pixel_groups: Array of PixelGroup structures (one per CUDA core)
    // group_pixel_coords: Flat array containing all pixel coordinates for all groups
    // group_pixel_counts: Array of counts, one per group (how many pixels in each group)
    // group_pixel_offsets: Array of offsets into group_pixel_coords for each group
    // num_groups: Total number of pixel groups (number of CUDA cores to use)
    void launch_render_kernel_custom(
        Color_cuda* d_image,
        int image_width,
        int image_height,
        Vec3_cuda camera_center,
        Vec3_cuda pixel100_loc,
        Vec3_cuda pixel_delta_u,
        Vec3_cuda pixel_delta_v,
        const Vertex_cuda* d_models,
        const int* d_model_triangle_counts,
        const int* d_model_triangle_offsets,
        int num_models,
        const Light_cuda* d_lights,
        int num_lights,
        const double* d_light_absorptions,
        int max_bounces,
        const PixelCoord_cuda* group_pixel_coords,
        const int* group_pixel_counts,
        const int* group_pixel_offsets,
        int num_groups
    );
}

// Helper function to prepare models for CUDA
void prepare_models_for_cuda(
    const std::vector<model>& models,
    std::vector<Vertex_cuda>& all_triangles,
    std::vector<int>& triangle_counts,
    std::vector<int>& triangle_offsets
);

#endif // RAY_TRACE_CUDA_H
