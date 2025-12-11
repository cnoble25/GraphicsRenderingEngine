#ifndef CAMERA_SETUP_H
#define CAMERA_SETUP_H

#include "vec3.h"
#include "ray_trace_cuda.h"
#include "constants.h"

// Camera parameters structure
struct CameraParams {
    Vec3_cuda camera_center;
    Vec3_cuda pixel100_loc;
    Vec3_cuda pixel_delta_u;
    Vec3_cuda pixel_delta_v;
};

/**
 * Setup camera parameters for rendering
 * @param image_width Width of output image in pixels
 * @param image_height Height of output image in pixels
 * @param focal_length Camera focal length (default: 1.0)
 * @param viewport_height Viewport height (default: 2.0)
 * @return CameraParams structure with CUDA-compatible camera parameters
 */
inline CameraParams setup_camera(int image_width, int image_height, 
                                 double focal_length = constants::DEFAULT_FOCAL_LENGTH, 
                                 double viewport_height = constants::DEFAULT_VIEWPORT_HEIGHT) {
    const auto aspect_ratio = static_cast<double>(image_width) / static_cast<double>(image_height);
    const auto viewport_width = viewport_height * aspect_ratio;
    const auto camera_center = point3(0, 0, 0);
    
    const auto viewport_u = vec3(viewport_width, 0, 0);
    const auto viewport_v = vec3(0, -viewport_height, 0);
    
    const auto pixel_delta_u = viewport_u / image_width;
    const auto pixel_delta_v = viewport_v / image_height;
    
    const auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) 
                                     - viewport_u/2 - viewport_v/2;
    const auto pixel100_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    
    CameraParams params;
    params.camera_center = vec3_to_cuda(camera_center);
    params.pixel100_loc = vec3_to_cuda(pixel100_loc);
    params.pixel_delta_u = vec3_to_cuda(pixel_delta_u);
    params.pixel_delta_v = vec3_to_cuda(pixel_delta_v);
    
    return params;
}

#endif // CAMERA_SETUP_H
