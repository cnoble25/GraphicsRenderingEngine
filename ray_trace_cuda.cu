#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <limits>

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
};

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
    
    return (P - O).magnitude();
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

// Ray color calculation
__device__ Color ray_color(const Ray& r, const Vertex* models, const int* model_triangle_counts, 
                           const int* model_triangle_offsets, int num_models, double luminosity) {
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
        return Color(luminosity / min_t, luminosity / min_t, luminosity / min_t);
    }
    
    // Background gradient
    Vec3 unit_dir = unit_vector(r.dir);
    double a = 0.5 * unit_dir.y + 1.0;
    return Color((1.0 - a) * 1.0 + a * 0.5,
                 (1.0 - a) * 1.0 + a * 0.7,
                 (1.0 - a) * 1.0 + a * 1.0);
}

// Main CUDA kernel for ray tracing
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
    double luminosity
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= image_width || j >= image_height) return;
    
    // Original code iterates: i from image_width down to 1, j from image_height down to 1
    // CUDA iterates: i from 0 to image_width-1, j from 0 to image_height-1
    // Map CUDA indices to original 1-based indices
    int original_i = image_width - i;  // CUDA i=0 -> original i=image_width, CUDA i=image_width-1 -> original i=1
    int original_j = image_height - j;  // CUDA j=0 -> original j=image_height, CUDA j=image_height-1 -> original j=1
    
    Vec3 pixel_center = pixel100_loc + pixel_delta_u * original_i + pixel_delta_v * original_j;
    Vec3 ray_direction = camera_center - pixel_center;
    Ray r(pixel_center, ray_direction);
    
    Color pixel_color = ray_color(r, models, model_triangle_counts, model_triangle_offsets, num_models, luminosity);
    
    // Store in array: CUDA j=0 corresponds to top row (original j=image_height)
    // When outputting, we'll output from j=image_height-1 down to 0 to match original order
    int idx = j * image_width + i;
    image[idx] = pixel_color;
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
    double luminosity
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((image_width + blockSize.x - 1) / blockSize.x,
                  (image_height + blockSize.y - 1) / blockSize.y);
    
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
        luminosity
    );
}
