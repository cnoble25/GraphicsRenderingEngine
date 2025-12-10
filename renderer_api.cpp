#include "renderer_api.h"
#include "model.h"
#include "transform.h"
#include "rotation.h"
#include "vec3.h"
#include "color.h"
#include "obj_loader.h"
#include "ray_trace_cuda.h"
#include "rasterization.h"
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

struct Scene {
    std::vector<model> objects;
    std::vector<std::string> obj_file_paths;  // Track OBJ file paths for reloading
};

extern "C" {

SceneHandle create_scene() {
    return new Scene();
}

int add_object_to_scene(SceneHandle scene, SceneObject_API* object) {
    if (!scene || !object) {
        return 0;  // Failure
    }
    
    Scene* s = static_cast<Scene*>(scene);
    
    transforms t(
        vec3(object->transform.position.x, object->transform.position.y, object->transform.position.z),
        rotations(object->transform.rotation.roll, object->transform.rotation.pitch, object->transform.rotation.yaw),
        vec3(object->transform.scale.x, object->transform.scale.y, object->transform.scale.z)
    );
    
    std::vector<vertex> base_vertices;
    
    switch (object->type) {
        case OBJECT_TYPE_PYRAMID: {
            model temp = pyamid();
            base_vertices = temp.vertices;
            break;
        }
            
        case OBJECT_TYPE_BOX: {
            model temp = box();
            base_vertices = temp.vertices;
            break;
        }
            
        case OBJECT_TYPE_OBJ_FILE:
            if (!object->obj_file_path) {
                return 0;  // Failure
            }
            {
                model temp = load_obj_file(std::string(object->obj_file_path));
                if (temp.vertices.empty()) {
                    return 0;  // Failed to load OBJ
                }
                base_vertices = temp.vertices;
            }
            s->obj_file_paths.push_back(std::string(object->obj_file_path));
            break;
            
        default:
            return 0;  // Unknown type
    }
    
    // Construct model directly and push it
    s->objects.emplace_back(base_vertices, t);
    return 1;  // Success
}

int remove_object_from_scene(SceneHandle scene, int index) {
    if (!scene) {
        return 0;
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->objects.size())) {
        return 0;
    }
    
    // Since model can't be moved/assigned, rebuild the vector without the element at index
    std::vector<model> new_objects;
    new_objects.reserve(s->objects.size() - 1);
    
    for (size_t i = 0; i < s->objects.size(); ++i) {
        if (i != static_cast<size_t>(index)) {
            new_objects.push_back(s->objects[i]);
        }
    }
    
    s->objects = std::move(new_objects);
    
    if (index < static_cast<int>(s->obj_file_paths.size())) {
        s->obj_file_paths.erase(s->obj_file_paths.begin() + index);
    }
    return 1;
}

int update_object_transform(SceneHandle scene, int index, Transform_API* transform) {
    if (!scene || !transform) {
        return 0;
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->objects.size())) {
        return 0;
    }
    
    // Get original vertices
    std::vector<vertex> original_vertices = s->objects[index].vertices;
    
    // Create new transform
    transforms t(
        vec3(transform->position.x, transform->position.y, transform->position.z),
        rotations(transform->rotation.roll, transform->rotation.pitch, transform->rotation.yaw),
        vec3(transform->scale.x, transform->scale.y, transform->scale.z)
    );
    
    // Since model can't be assigned, we need to rebuild the vector
    // Store all objects, erase the old one, and rebuild
    std::vector<model> new_objects;
    new_objects.reserve(s->objects.size());
    
    for (size_t i = 0; i < s->objects.size(); ++i) {
        if (i == static_cast<size_t>(index)) {
            // Create new model with updated transform
            new_objects.emplace_back(original_vertices, t);
        } else {
            // Copy existing model (copy constructor should work)
            new_objects.push_back(s->objects[i]);
        }
    }
    
    s->objects = std::move(new_objects);
    
    return 1;
}

int get_scene_object_count(SceneHandle scene) {
    if (!scene) {
        return 0;
    }
    Scene* s = static_cast<Scene*>(scene);
    return static_cast<int>(s->objects.size());
}

int render_scene(SceneHandle scene, const char* output_path, int image_width, int image_height, double luminosity, RenderMode_API render_mode) {
    if (!scene || !output_path) {
        return 0;
    }
    
    Scene* s = static_cast<Scene*>(scene);
    
    if (s->objects.empty()) {
        std::cerr << "Scene is empty!" << std::endl;
        return 0;
    }
    
    std::vector<color> image(image_width * image_height);
    
    if (render_mode == RENDER_MODE_RASTERIZATION) {
        // Rasterization path
        rasterize_scene(s->objects, image, image_width, image_height, luminosity);
    } else {
        // Ray tracing path (CUDA)
        // Initialize CUDA
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 0;
        }
        
        // Prepare models for CUDA
        std::vector<Vertex_cuda> all_triangles;
        std::vector<int> triangle_counts;
        std::vector<int> triangle_offsets;
        prepare_models_for_cuda(s->objects, all_triangles, triangle_counts, triangle_offsets);
        
        if (all_triangles.empty()) {
            std::cerr << "No triangles in scene!" << std::endl;
            return 0;
        }
        
        // Camera setup
        auto aspect_ratio = static_cast<double>(image_width) / static_cast<double>(image_height);
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * aspect_ratio;
        auto camera_center = point3(0, 0, 0);
        
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);
        
        auto pixel_delta_u = viewport_u / image_width;
        auto pixel_delta_v = viewport_v / image_height;
        
        auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        auto pixel100_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        
        // Convert to CUDA format
        Vec3_cuda cuda_camera_center = vec3_to_cuda(camera_center);
        Vec3_cuda cuda_pixel100_loc = vec3_to_cuda(pixel100_loc);
        Vec3_cuda cuda_pixel_delta_u = vec3_to_cuda(pixel_delta_u);
        Vec3_cuda cuda_pixel_delta_v = vec3_to_cuda(pixel_delta_v);
        
        // Allocate device memory
        Vertex_cuda* d_models = nullptr;
        int* d_triangle_counts = nullptr;
        int* d_triangle_offsets = nullptr;
        Color_cuda* d_image = nullptr;
        
        size_t triangles_size = all_triangles.size() * sizeof(Vertex_cuda);
        size_t counts_size = triangle_counts.size() * sizeof(int);
        size_t offsets_size = triangle_offsets.size() * sizeof(int);
        size_t image_size = image_width * image_height * sizeof(Color_cuda);
        
        std::vector<Color_cuda> cuda_image(image_width * image_height);
        
        cudaStatus = cudaMalloc((void**)&d_models, triangles_size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_models: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        cudaStatus = cudaMalloc((void**)&d_triangle_counts, counts_size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_triangle_counts: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        cudaStatus = cudaMalloc((void**)&d_triangle_offsets, offsets_size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_triangle_offsets: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        cudaStatus = cudaMalloc((void**)&d_image, image_size);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_image: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        // Copy data to device
        cudaStatus = cudaMemcpy(d_models, all_triangles.data(), triangles_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for d_models: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        cudaStatus = cudaMemcpy(d_triangle_counts, triangle_counts.data(), counts_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for d_triangle_counts: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        cudaStatus = cudaMemcpy(d_triangle_offsets, triangle_offsets.data(), offsets_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for d_triangle_offsets: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        // Launch kernel
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
            s->objects.size(),
            luminosity
        );
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        // Copy result back
        cudaStatus = cudaMemcpy(cuda_image.data(), d_image, image_size, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for image: " << cudaGetErrorString(cudaStatus) << std::endl;
            goto cleanup;
        }
        
        // Convert CUDA colors to regular colors
        for (int i = 0; i < image_width * image_height; ++i) {
            image[i] = cuda_to_color(cuda_image[i]);
        }
        
        // Free CUDA memory (successful path)
        if (d_models) cudaFree(d_models);
        if (d_triangle_counts) cudaFree(d_triangle_counts);
        if (d_triangle_offsets) cudaFree(d_triangle_offsets);
        if (d_image) cudaFree(d_image);
        
        // Clear pointers to prevent double-free
        d_models = nullptr;
        d_triangle_counts = nullptr;
        d_triangle_offsets = nullptr;
        d_image = nullptr;
        
cleanup:
        // Cleanup on error only
        if (cudaStatus != cudaSuccess) {
            if (d_models) cudaFree(d_models);
            if (d_triangle_counts) cudaFree(d_triangle_counts);
            if (d_triangle_offsets) cudaFree(d_triangle_offsets);
            if (d_image) cudaFree(d_image);
            return 0;
        }
    }
    
    // Write PPM file
    {
        std::ofstream outfile(output_path, std::ios::out | std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file: " << output_path << std::endl;
            return 0;
        }
        
        outfile << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        outfile.flush();
        
        for (int j = 0; j < image_height; j++) {
            for (int i = image_width - 1; i >= 0; i--) {
                int idx = j * image_width + i;
                write_color(outfile, image[idx]);
            }
        }
        
        outfile.flush();
        outfile.close();
        
        // Verify file was written
        std::ifstream verify(output_path);
        if (!verify.good()) {
            std::cerr << "Failed to verify output file: " << output_path << std::endl;
            return 0;
        }
        verify.close();
    }
    
    return 1;
}

void free_scene(SceneHandle scene) {
    if (scene) {
        delete static_cast<Scene*>(scene);
    }
}

} // extern "C"
