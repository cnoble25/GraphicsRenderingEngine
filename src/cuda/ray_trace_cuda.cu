#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <limits>
#include <algorithm>

// CUDA-compatible vec3 structure
struct Vec3 {
    double x, y, z;
    
    __device__ Vec3() : x(0), y(0), z(0) {}
    __device__ Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    
    __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    
    __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    
    __device__ Vec3 operator*(double t) const {
        return Vec3(x * t, y * t, z * t);
    }
    
    __device__ Vec3 operator/(double t) const {
        return Vec3(x / t, y / t, z / t);
    }
    
    __device__ double magnitude() const {
        return sqrt(x*x + y*y + z*z);
    }
    
    __device__ double length_squared() const {
        return x*x + y*y + z*z;
    }
};

// CUDA-compatible ray structure
struct Ray {
    Vec3 orig;
    Vec3 dir;
    
    __device__ Ray() {}
    __device__ Ray(const Vec3& origin, const Vec3& direction) : orig(origin), dir(direction) {}
    
    __device__ Vec3 at(double t) const {
        return orig + dir * t;
    }
};

// CUDA-compatible vertex structure (triangle)
struct Vertex {
    Vec3 p0, p1, p2;  // Three points of the triangle
    
    __device__ Vertex() {}
    __device__ Vertex(const Vec3& v0, const Vec3& v1, const Vec3& v2) : p0(v0), p1(v1), p2(v2) {}
};

// CUDA-compatible color structure
struct Color {
    double r, g, b;
    
    __device__ Color() : r(0), g(0), b(0) {}
    __device__ Color(double r_, double g_, double b_) : r(r_), g(g_), b(b_) {}
    
    __device__ Color operator+(const Color& c) const {
        return Color(r + c.r, g + c.g, b + c.b);
    }
    
    __device__ Color operator*(double t) const {
        return Color(r * t, g * t, b * t);
    }
};

// CUDA-compatible light structure
struct Light {
    Vec3 position;
    Color color;
    double luminosity;
    
    __device__ Light() : position(Vec3(0, 0, 0)), color(Color(1, 1, 1)), luminosity(1.0) {}
    __device__ Light(const Vec3& pos, const Color& col, double lum) 
        : position(pos), color(col), luminosity(lum) {}
};

// PixelCoord_cuda is defined in ray_trace_cuda.h

// Device-side utility functions
__device__ double dot(const Vec3& u, const Vec3& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

__device__ Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

__device__ Vec3 unit_vector(const Vec3& v) {
    double mag = v.magnitude();
    if (mag < 1e-8) return Vec3(0, 0, 0);
    return v / mag;
}

// Ray-triangle intersection test
__device__ double ray_triangle_intersect(const Ray& r, const Vertex& tri) {
    // Compute the plane's normal
    Vec3 v0v1 = tri.p1 - tri.p0;
    Vec3 v0v2 = tri.p2 - tri.p0;
    Vec3 N = cross(v0v1, v0v2);
    
    // Check if the ray and plane are parallel
    double NdotRayDirection = dot(N, r.dir);
    if (fabs(NdotRayDirection) < 1e-8) {
        return -1.0; // They are parallel, so they don't intersect!
    }
    
    // Compute d parameter
    double d = -dot(N, tri.p0);
    
    // Compute t
    double t = -(dot(N, r.orig) + d) / NdotRayDirection;
    
    // Check if the triangle is behind the ray
    if (t < 0) return -1.0;
    
    // Compute the intersection point
    Vec3 P = r.orig + r.dir * t;
    
    Vec3 O = r.orig + r.dir;
    
    // Step 2: Inside-Outside Test
    Vec3 Ne;
    
    // Test sidedness of P w.r.t. edge v0v1
    Vec3 v0p = P - tri.p0;
    Ne = cross(v0v1, v0p);
    if (dot(N, Ne) < 0) return -1.0;
    
    // Test sidedness of P w.r.t. edge v2v1
    Vec3 v2v1 = tri.p2 - tri.p1;
    Vec3 v1p = P - tri.p1;
    Ne = cross(v2v1, v1p);
    if (dot(N, Ne) < 0) return -1.0;
    
    // Test sidedness of P w.r.t. edge v2v0
    Vec3 v2v0 = tri.p0 - tri.p2;
    Vec3 v2p = P - tri.p2;
    Ne = cross(v2v0, v2p);
    if (dot(N, Ne) < 0) return -1.0;
    
    // Return the t parameter (distance along the ray)
    return t;
}

// Find closest intersection with all triangles in a model
__device__ double model_intersect(const Ray& r, const Vertex* triangles, int num_triangles) {
    double min_t = -1.0;
    
    for (int i = 0; i < num_triangles; i++) {
        double t = ray_triangle_intersect(r, triangles[i]);
        if (t > 0) {
            if (min_t < 0 || t < min_t) {
                min_t = t;
            }
        }
    }
    
    return min_t;
}

// Check if a shadow ray hits any objects before reaching the light
// Returns true if the light is blocked (shadow), false if light reaches the point
__device__ bool is_in_shadow(const Vec3& point, const Vec3& light_pos, const Vec3& normal,
                             const Vertex* models, const int* model_triangle_counts,
                             const int* model_triangle_offsets, int num_models) {
    // Create shadow ray from point towards light
    // Offset the ray origin slightly along the normal to avoid self-intersection
    // Use a larger offset to ensure we don't hit the same surface
    Vec3 offset_point = point + normal * 1e-3;  // Offset to avoid hitting the same surface
    Vec3 to_light = light_pos - offset_point;
    double distance_to_light = to_light.magnitude();
    
    if (distance_to_light < 1e-5) {
        return false;  // Light is at the point, not shadowed
    }
    
    Vec3 light_dir = unit_vector(to_light);
    Ray shadow_ray(offset_point, light_dir);
    
    // Check for intersections with all models
    // We want to find if anything blocks the path to the light
    for (int model_idx = 0; model_idx < num_models; model_idx++) {
        int offset = model_triangle_offsets[model_idx];
        int count = model_triangle_counts[model_idx];
        
        double t = model_intersect(shadow_ray, &models[offset], count);
        // Check if intersection is between the offset point and the light
        // Use a small tolerance to account for floating point precision
        if (t > 1e-5 && t < distance_to_light - 1e-3) {
            return true;  // In shadow - something is blocking the light
        }
    }
    
    return false;  // Not in shadow, light reaches the point
}

// Get surface normal at intersection point
__device__ Vec3 get_surface_normal(const Ray& r, double t, const Vertex* models, 
                                    const int* model_triangle_counts, const int* model_triangle_offsets, 
                                    int num_models, int& hit_model_idx, int& hit_triangle_idx) {
    double closest_t = -1.0;
    hit_model_idx = -1;
    hit_triangle_idx = -1;
    
    // Find which triangle was hit
    for (int model_idx = 0; model_idx < num_models; model_idx++) {
        int offset = model_triangle_offsets[model_idx];
        int count = model_triangle_counts[model_idx];
        
        for (int i = 0; i < count; i++) {
            double tri_t = ray_triangle_intersect(r, models[offset + i]);
            if (tri_t > 0 && (closest_t < 0 || tri_t < closest_t)) {
                closest_t = tri_t;
                hit_model_idx = model_idx;
                hit_triangle_idx = offset + i;
            }
        }
    }
    
    if (hit_triangle_idx >= 0) {
        const Vertex& tri = models[hit_triangle_idx];
        Vec3 v0v1 = tri.p1 - tri.p0;
        Vec3 v0v2 = tri.p2 - tri.p0;
        Vec3 normal = cross(v0v1, v0v2);
        normal = unit_vector(normal);
        
        // Make sure normal points towards the camera (flip if needed)
        // The ray direction points from pixel to camera, so we negate it
        Vec3 neg_dir = r.dir * -1.0;
        Vec3 view_dir = unit_vector(neg_dir);
        if (dot(normal, view_dir) < 0) {
            normal = normal * -1.0;  // Flip normal
        }
        
        return normal;
    }
    
    return Vec3(0, 1, 0);  // Default normal (up)
}

// Calculate reflection direction: R = I - 2 * (I · N) * N
// where I is the incident direction and N is the surface normal
__device__ Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - normal * (2.0 * dot(incident, normal));
}

// Ray color calculation with proper lighting from light sources and reflections
// incoming_light_energy: Light energy carried by this ray (from previous bounces)
__device__ Color ray_color(const Ray& r, const Vertex* models, const int* model_triangle_counts, 
                           const int* model_triangle_offsets, int num_models, 
                           const Light* lights, int num_lights,
                           const double* light_absorptions, int max_bounces, int current_bounce,
                           Color incoming_light_energy = Color(0, 0, 0)) {
    double min_t = -1.0;
    
    // Check intersection with all models
    for (int model_idx = 0; model_idx < num_models; model_idx++) {
        int offset = model_triangle_offsets[model_idx];
        int count = model_triangle_counts[model_idx];
        
        double t = model_intersect(r, &models[offset], count);
        if (t > 0) {
            if (min_t < 0 || t < min_t) {
                min_t = t;
            }
        }
    }
    
    if (min_t > -0.5) {
        // Calculate intersection point
        Vec3 intersection_point = r.at(min_t);
        
        // Get surface normal at intersection
        int hit_model_idx, hit_triangle_idx;
        Vec3 normal = get_surface_normal(r, min_t, models, model_triangle_counts, 
                                          model_triangle_offsets, num_models, 
                                          hit_model_idx, hit_triangle_idx);
        
        // Get light absorption for this object (default to 1.0 if invalid index, meaning completely matte)
        double object_light_absorption = 1.0;
        if (hit_model_idx >= 0 && hit_model_idx < num_models) {
            if (light_absorptions != nullptr) {
                object_light_absorption = light_absorptions[hit_model_idx];
                // Clamp light absorption to [0, 1]
                object_light_absorption = fmax(0.0, fmin(1.0, object_light_absorption));
            }
        }
        
        // Calculate reflectivity from light absorption: reflectivity = 1.0 - absorption
        // 0.0 absorption = perfect mirror (1.0 reflectivity)
        // 1.0 absorption = completely matte (0.0 reflectivity)
        double object_reflectivity = 1.0 - object_light_absorption;
        
        // Start with incoming light energy (from previous bounces)
        // This is light that has bounced and is now hitting this surface
        Color total_light = incoming_light_energy;
        
        // Calculate lighting from all light sources
        for (int light_idx = 0; light_idx < num_lights; light_idx++) {
            const Light& light = lights[light_idx];
            
            // Vector from intersection point to light
            Vec3 to_light = light.position - intersection_point;
            double distance_to_light = to_light.magnitude();
            
            // Skip if light is too close (avoid numerical issues)
            if (distance_to_light < 1e-6) continue;
            
            // Check if this point is in shadow from this light
            bool in_shadow = is_in_shadow(intersection_point, light.position, normal,
                                         models, model_triangle_counts, model_triangle_offsets, num_models);
            
            // If in shadow, skip this light (don't add its contribution)
            if (in_shadow) continue;
            
            Vec3 light_dir = unit_vector(to_light);
            
            // Calculate light contribution using inverse square law
            // Scale luminosity significantly to make lights clearly visible
            double distance_squared = distance_to_light * distance_to_light;
            // Use larger scaling factor - luminosity of 5.0 should be clearly visible
            double light_intensity = (light.luminosity * 100.0) / (distance_squared + 0.5);
            
            // Lambertian shading: dot product of normal and light direction
            double cos_angle = dot(normal, light_dir);
            cos_angle = fmax(0.0, cos_angle);  // Clamp to [0, 1] - no negative lighting
            
            // Apply light color and intensity
            // Don't scale by (1 - reflectivity) here - we want full direct lighting
            // Reflections will add additional light on top
            Color light_contribution = light.color * (light_intensity * cos_angle);
            total_light = total_light + light_contribution;
        }
        
        // Add very minimal ambient lighting ONLY so completely unlit areas aren't pure black
        // Don't scale ambient by reflectivity - reflections add to the total
        Color ambient = Color(0.02, 0.02, 0.02);
        total_light = total_light + ambient;
        
        // Handle reflections if reflectivity > 0 and we haven't exceeded max bounces
        if (object_reflectivity > 1e-6 && current_bounce < max_bounces) {
            // Calculate reflection direction
            // The ray direction (r.dir) points from pixel to camera
            // For reflection, we need the incident direction pointing FROM the intersection point TO the camera
            // This is the direction the light came from (view direction)
            Vec3 view_dir = unit_vector(r.dir * -1.0);  // Negate to get direction from intersection to camera
            
            // Calculate reflected direction using: R = I - 2 * (I · N) * N
            // This gives us the direction the reflected ray should travel
            Vec3 reflected_dir = reflect(view_dir, normal);
            
            // Offset the ray origin slightly along the normal to avoid self-intersection
            Vec3 offset_origin = intersection_point + normal * 1e-3;
            
            // Create reflected ray - direction points away from surface
            Ray reflected_ray(offset_origin, reflected_dir);
            
            // Calculate the light energy that will be reflected (carried by the reflected ray)
            // This is the total light at this surface scaled by reflectivity
            Color reflected_light_energy = total_light * object_reflectivity;
            
            // Recursively trace the reflected ray, passing the light energy it carries
            // The next surface will receive this light energy and add it to its own lighting
            Color reflected_color = ray_color(reflected_ray, models, model_triangle_counts,
                                              model_triangle_offsets, num_models,
                                              lights, num_lights,
                                              light_absorptions, max_bounces, current_bounce + 1,
                                              reflected_light_energy);
            
            // The reflected color now includes:
            // 1. Direct lighting at the next surface
            // 2. Incoming light energy from this surface (reflected_light_energy)
            // 3. Any further bounces from that surface
            
            // Add the reflected contribution to this surface's lighting
            // The reflected light is already scaled by the next surface's properties
            total_light = total_light * (1.0 - object_reflectivity) + reflected_color;
        }
        
        // Apply simple tone mapping - less aggressive to preserve brightness
        double max_component = fmax(total_light.r, fmax(total_light.g, total_light.b));
        if (max_component > 1.0) {
            // Use Reinhard tone mapping: x / (1 + x) but less aggressive
            double scale = 1.0 / (1.0 + (max_component - 1.0) * 0.5);
            total_light = total_light * scale;
        }
        
        // Clamp to valid range
        total_light.r = fmin(1.0, fmax(0.0, total_light.r));
        total_light.g = fmin(1.0, fmax(0.0, total_light.g));
        total_light.b = fmin(1.0, fmax(0.0, total_light.b));
        
        return total_light;
    }
    
    // Background gradient
    Vec3 unit_dir = unit_vector(r.dir);
    double a = 0.5 * unit_dir.y + 1.0;
    return Color((1.0 - a) * 1.0 + a * 0.5,
                 (1.0 - a) * 1.0 + a * 0.7,
                 (1.0 - a) * 1.0 + a * 1.0);
}

// ============================================================================
// PIXEL PROCESSING FUNCTION - MODIFY THIS TO CHANGE PIXEL OUTPUT
// ============================================================================
// This function is called for each pixel after ray tracing.
// You can modify this function to apply post-processing effects, filters,
// color grading, or any other pixel-level manipulations.
//
// Parameters:
//   - pixel_color: The color computed by ray tracing
//   - x, y: Pixel coordinates (0 to image_width-1, 0 to image_height-1)
//   - image_width, image_height: Image dimensions
//
// Returns: The final color that will be displayed for this pixel
// ============================================================================
__device__ Color process_pixel(const Color& pixel_color, int x, int y, int image_width, int image_height) {
    // Default implementation: return the ray-traced color as-is
    // Modify this function to add your custom pixel effects!
    
    // Example: Uncomment to invert colors
    // return Color(1.0 - pixel_color.r, 1.0 - pixel_color.g, 1.0 - pixel_color.b);
    
    // Example: Uncomment to add a vignette effect
    // double center_x = image_width / 2.0;
    // double center_y = image_height / 2.0;
    // double dist_x = (x - center_x) / center_x;
    // double dist_y = (y - center_y) / center_y;
    // double dist = sqrt(dist_x * dist_x + dist_y * dist_y);
    // double vignette = 1.0 - dist * 0.5;
    // vignette = fmax(0.0, fmin(1.0, vignette));
    // return Color(pixel_color.r * vignette, pixel_color.g * vignette, pixel_color.b * vignette);
    
    // Example: Uncomment to apply sepia tone
    // double r = pixel_color.r * 0.393 + pixel_color.g * 0.769 + pixel_color.b * 0.189;
    // double g = pixel_color.r * 0.349 + pixel_color.g * 0.686 + pixel_color.b * 0.168;
    // double b = pixel_color.r * 0.272 + pixel_color.g * 0.534 + pixel_color.b * 0.131;
    // return Color(fmin(1.0, r), fmin(1.0, g), fmin(1.0, b));
    
    // Return original color (no modification)
    return pixel_color;
}

// Main CUDA kernel for ray tracing with block-based compression
__global__ void render_kernel(
    Color* image,
    int image_width,
    int image_height,
    Vec3 camera_center,
    Vec3 pixel100_loc,
    Vec3 pixel_delta_u,
    Vec3 pixel_delta_v,
    const Vertex* models,
    const int* model_triangle_counts,
    const int* model_triangle_offsets,
    int num_models,
    const Light* lights,
    int num_lights,
    const double* light_absorptions,
    int max_bounces,
    int compression_level
) {
    // Clamp compression_level to valid range [1, max(image_width, image_height)]
    compression_level = (compression_level < 1) ? 1 : compression_level;
    
    // Calculate number of blocks in each dimension
    int blocks_x = (image_width + compression_level - 1) / compression_level;
    int blocks_y = (image_height + compression_level - 1) / compression_level;
    
    // Get block index (which block this thread processes)
    int block_i = blockIdx.x * blockDim.x + threadIdx.x;
    int block_j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if this thread handles a valid block
    if (block_i >= blocks_x || block_j >= blocks_y) return;
    
    // Calculate the top-left pixel of this block
    int start_i = block_i * compression_level;
    int start_j = block_j * compression_level;
    
    // Calculate center pixel of the block for ray tracing
    int center_i = start_i + compression_level / 2;
    int center_j = start_j + compression_level / 2;
    
    // Clamp center to image bounds
    center_i = (center_i > image_width - 1) ? (image_width - 1) : center_i;
    center_j = (center_j > image_height - 1) ? (image_height - 1) : center_j;
    
    // Map CUDA indices to original 1-based indices for center pixel
    int original_i = image_width - center_i;
    int original_j = image_height - center_j;
    
    // Compute ray for the center pixel of the block
    Vec3 pixel_center = pixel100_loc + pixel_delta_u * original_i + pixel_delta_v * original_j;
    Vec3 ray_direction = camera_center - pixel_center;
    Ray r(pixel_center, ray_direction);
    
    // Initial ray carries no light energy (it's coming from the camera)
    Color block_color = ray_color(r, models, model_triangle_counts, model_triangle_offsets, num_models, lights, num_lights, light_absorptions, max_bounces, 0, Color(0, 0, 0));
    
    // Process pixel through user-defined pixel manipulation function
    Color final_color = process_pixel(block_color, center_i, center_j, image_width, image_height);
    
    // Output which pixels this CUDA thread/core is processing
    // Print all blocks to show compression grouping
    printf("[CUDA Core: block(%d,%d) thread(%d,%d)] Block[%d,%d] processes pixels: ", 
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, block_i, block_j);
    
    // List all pixels in this block
    int pixel_count = 0;
    for (int dy = 0; dy < compression_level; dy++) {
        for (int dx = 0; dx < compression_level; dx++) {
            int pixel_i = start_i + dx;
            int pixel_j = start_j + dy;
            if (pixel_i < image_width && pixel_j < image_height) {
                if (pixel_count > 0) printf(", ");
                printf("(%d,%d)", pixel_i, pixel_j);
                pixel_count++;
            }
        }
    }
    printf(" -> All assigned RGB(%.3f,%.3f,%.3f) [%d pixels share same color]\n", 
           final_color.r, final_color.g, final_color.b, pixel_count);
    
    // Write the same color to all pixels in this block
    for (int dy = 0; dy < compression_level; dy++) {
        for (int dx = 0; dx < compression_level; dx++) {
            int pixel_i = start_i + dx;
            int pixel_j = start_j + dy;
            
            // Check bounds
            if (pixel_i < image_width && pixel_j < image_height) {
                int idx = pixel_j * image_width + pixel_i;
                image[idx] = final_color;
            }
        }
    }
}

// Custom compression kernel - each CUDA core processes a custom set of pixels
// Ray is calculated from the average position of all pixels in the group
__global__ void render_kernel_custom(
    Color* image,
    int image_width,
    int image_height,
    Vec3 camera_center,
    Vec3 pixel100_loc,
    Vec3 pixel_delta_u,
    Vec3 pixel_delta_v,
    const Vertex* models,
    const int* model_triangle_counts,
    const int* model_triangle_offsets,
    int num_models,
    const Light* lights,
    int num_lights,
    const double* light_absorptions,
    int max_bounces,
    const PixelCoord_cuda* group_pixel_coords,
    const int* group_pixel_counts,
    const int* group_pixel_offsets,
    int num_groups
) {
    // Get which pixel group this thread processes
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread handles a valid group
    if (group_idx >= num_groups) return;
    
    // Get pixel count and offset for this group
    int pixel_count = group_pixel_counts[group_idx];
    int offset = group_pixel_offsets[group_idx];
    
    if (pixel_count <= 0) return;  // Skip empty groups
    
    // Calculate average position of all pixels in this group
    double avg_x = 0.0;
    double avg_y = 0.0;
    int valid_pixels = 0;
    
    for (int i = 0; i < pixel_count; i++) {
        PixelCoord_cuda coord = group_pixel_coords[offset + i];
        // Validate pixel coordinates
        if (coord.x >= 0 && coord.x < image_width && coord.y >= 0 && coord.y < image_height) {
            avg_x += coord.x;
            avg_y += coord.y;
            valid_pixels++;
        }
    }
    
    if (valid_pixels == 0) return;  // No valid pixels in this group
    
    // Calculate average
    avg_x /= valid_pixels;
    avg_y /= valid_pixels;
    
    // Round to nearest pixel for ray calculation
    int avg_pixel_i = (int)(avg_x + 0.5);
    int avg_pixel_j = (int)(avg_y + 0.5);
    
    // Clamp to image bounds
    avg_pixel_i = (avg_pixel_i > image_width - 1) ? (image_width - 1) : avg_pixel_i;
    avg_pixel_j = (avg_pixel_j > image_height - 1) ? (image_height - 1) : avg_pixel_j;
    avg_pixel_i = (avg_pixel_i < 0) ? 0 : avg_pixel_i;
    avg_pixel_j = (avg_pixel_j < 0) ? 0 : avg_pixel_j;
    
    // Map CUDA indices to original 1-based indices for average pixel
    int original_i = image_width - avg_pixel_i;
    int original_j = image_height - avg_pixel_j;
    
    // Compute ray for the average pixel position
    Vec3 pixel_center = pixel100_loc + pixel_delta_u * original_i + pixel_delta_v * original_j;
    Vec3 ray_direction = camera_center - pixel_center;
    Ray r(pixel_center, ray_direction);
    
    // Initial ray carries no light energy (it's coming from the camera)
    Color group_color = ray_color(r, models, model_triangle_counts, model_triangle_offsets, num_models, lights, num_lights, light_absorptions, max_bounces, 0, Color(0, 0, 0));
    
    // Process pixel through user-defined pixel manipulation function
    Color final_color = process_pixel(group_color, avg_pixel_i, avg_pixel_j, image_width, image_height);
    
    // Output which pixels this CUDA thread/core is processing
    printf("[CUDA Core: block(%d,%d) thread(%d,%d)] Group[%d] processes pixels: ", 
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, group_idx);
    
    // List all pixels in this group
    for (int i = 0; i < pixel_count; i++) {
        PixelCoord_cuda coord = group_pixel_coords[offset + i];
        if (coord.x >= 0 && coord.x < image_width && coord.y >= 0 && coord.y < image_height) {
            if (i > 0) printf(", ");
            printf("(%d,%d)", coord.x, coord.y);
        }
    }
    printf(" -> Average pos: (%.2f,%.2f) -> RGB(%.3f,%.3f,%.3f) [%d pixels share same color]\n", 
           avg_x, avg_y, final_color.r, final_color.g, final_color.b, valid_pixels);
    
    // Write the same color to all pixels in this group
    for (int i = 0; i < pixel_count; i++) {
        PixelCoord_cuda coord = group_pixel_coords[offset + i];
        if (coord.x >= 0 && coord.x < image_width && coord.y >= 0 && coord.y < image_height) {
            int idx = coord.y * image_width + coord.x;
            image[idx] = final_color;
        }
    }
}

// Host function to launch the kernel
extern "C" void launch_render_kernel(
    Color* d_image,
    int image_width,
    int image_height,
    Vec3 camera_center,
    Vec3 pixel100_loc,
    Vec3 pixel_delta_u,
    Vec3 pixel_delta_v,
    const Vertex* d_models,
    const int* d_model_triangle_counts,
    const int* d_model_triangle_offsets,
    int num_models,
    const Light* d_lights,
    int num_lights,
    const double* d_light_absorptions,
    int max_bounces,
    int compression_level
) {
    // Clamp compression_level to valid range
    compression_level = std::max(1, compression_level);
    
    // Calculate number of blocks in each dimension
    int blocks_x = (image_width + compression_level - 1) / compression_level;
    int blocks_y = (image_height + compression_level - 1) / compression_level;
    
    // Use 16x16 thread blocks
    dim3 blockSize(16, 16);
    // Grid size is based on number of blocks, not pixels
    dim3 gridSize((blocks_x + blockSize.x - 1) / blockSize.x,
                  (blocks_y + blockSize.y - 1) / blockSize.y);
    
    render_kernel<<<gridSize, blockSize>>>(
        d_image,
        image_width,
        image_height,
        camera_center,
        pixel100_loc,
        pixel_delta_u,
        pixel_delta_v,
        d_models,
        d_model_triangle_counts,
        d_model_triangle_offsets,
        num_models,
        d_lights,
        num_lights,
        d_light_absorptions,
        max_bounces,
        compression_level
    );
}

// Host function to launch the custom compression kernel
extern "C" void launch_render_kernel_custom(
    Color* d_image,
    int image_width,
    int image_height,
    Vec3 camera_center,
    Vec3 pixel100_loc,
    Vec3 pixel_delta_u,
    Vec3 pixel_delta_v,
    const Vertex* d_models,
    const int* d_model_triangle_counts,
    const int* d_model_triangle_offsets,
    int num_models,
    const Light* d_lights,
    int num_lights,
    const double* d_light_absorptions,
    int max_bounces,
    const PixelCoord_cuda* group_pixel_coords,
    const int* group_pixel_counts,
    const int* group_pixel_offsets,
    int num_groups
) {
    // Use 256 threads per block (1D grid)
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_groups + threadsPerBlock - 1) / threadsPerBlock;
    
    render_kernel_custom<<<blocksPerGrid, threadsPerBlock>>>(
        d_image,
        image_width,
        image_height,
        camera_center,
        pixel100_loc,
        pixel_delta_u,
        pixel_delta_v,
        d_models,
        d_model_triangle_counts,
        d_model_triangle_offsets,
        num_models,
        d_lights,
        num_lights,
        d_light_absorptions,
        max_bounces,
        group_pixel_coords,
        group_pixel_counts,
        group_pixel_offsets,
        num_groups
    );
}
