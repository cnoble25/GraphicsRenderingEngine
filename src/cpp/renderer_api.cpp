// Corresponding header
#include "renderer_api.h"
// System headers
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#endif
// Third-party headers
#include <cuda_runtime.h>
// Project headers
#include "model.h"
#include "transform.h"
#include "rotation.h"
#include "vec3.h"
#include "color.h"
#include "obj_loader.h"
#include "ray_trace_cuda.h"
#include "light.h"
#include "errors.h"
#include "cuda_memory.h"
#include "cuda_utils.h"
#include "camera_setup.h"
#include "constants.h"

struct Scene {
    std::vector<model> objects;
    std::vector<std::string> obj_file_paths;  // Track OBJ file paths for reloading
    std::vector<Light> lights;  // Scene lights
};

extern "C" {

SceneHandle create_scene() {
    return new Scene();
}

int add_object_to_scene(SceneHandle scene, SceneObject_API* object) {
    if (!scene || !object) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
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
            model temp = pyramid();
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
                return static_cast<int>(RendererError::OBJ_FILE_NOT_FOUND);
            }
            {
                model temp = load_obj_file(std::string(object->obj_file_path));
                if (temp.vertices.empty()) {
                    return static_cast<int>(RendererError::OBJ_FILE_LOAD_FAILED);
                }
                base_vertices = temp.vertices;
            }
            s->obj_file_paths.push_back(std::string(object->obj_file_path));
            break;
            
        default:
            return static_cast<int>(RendererError::INVALID_OBJECT);
    }
    
    // Construct model directly and push it
    s->objects.emplace_back(base_vertices, t);
    return static_cast<int>(RendererError::SUCCESS);
}

int remove_object_from_scene(SceneHandle scene, int index) {
    if (!scene) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->objects.size())) {
        return static_cast<int>(RendererError::INVALID_INDEX);
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
    return static_cast<int>(RendererError::SUCCESS);
}

int update_object_transform(SceneHandle scene, int index, Transform_API* transform) {
    if (!scene || !transform) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->objects.size())) {
        return static_cast<int>(RendererError::INVALID_INDEX);
    }
    
    // Create new transform
    const transforms t(
        vec3(transform->position.x, transform->position.y, transform->position.z),
        rotations(transform->rotation.roll, transform->rotation.pitch, transform->rotation.yaw),
        vec3(transform->scale.x, transform->scale.y, transform->scale.z)
    );
    
    // Update transform in-place using new method (no vector rebuild needed!)
    s->objects[index].update_transform(t);
    
    return static_cast<int>(RendererError::SUCCESS);
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
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    
    if (s->objects.empty()) {
        std::cerr << "Scene is empty!" << std::endl;
        return static_cast<int>(RendererError::EMPTY_SCENE);
    }
    
    // Ensure output directory exists
    std::string output_str(output_path);
    try {
        std::filesystem::path file_path(output_str);
        std::filesystem::path dir_path = file_path.parent_path();
        if (!dir_path.empty()) {
            std::filesystem::create_directories(dir_path);
        }
    } catch (const std::exception& e) {
        // If filesystem fails, try fallback method
        size_t last_slash = output_str.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            std::string dir_path = output_str.substr(0, last_slash);
#if !defined(_WIN32) && !defined(_WIN64)
            mkdir(dir_path.c_str(), 0755);
#else
            _mkdir(dir_path.c_str());
#endif
        }
    }
    
    std::vector<color> image(image_width * image_height);
    
    // Ray tracing path (CUDA) - only rendering mode
    try {
        // Initialize CUDA using utility function
        cuda_utils::set_device(0);
        
        // Prepare models for CUDA
        std::vector<Vertex_cuda> all_triangles;
        std::vector<int> triangle_counts;
        std::vector<int> triangle_offsets;
        prepare_models_for_cuda(s->objects, all_triangles, triangle_counts, triangle_offsets);
        
        if (all_triangles.empty()) {
            std::cerr << "No triangles in scene!" << std::endl;
            return static_cast<int>(RendererError::NO_TRIANGLES_IN_SCENE);
        }
        
        // Camera setup using helper function
        const CameraParams camera = setup_camera(image_width, image_height);
        
        // Allocate device memory using RAII
        CudaBuffer<Vertex_cuda> d_models(all_triangles.size());
        CudaBuffer<int> d_triangle_counts(triangle_counts.size());
        CudaBuffer<int> d_triangle_offsets(triangle_offsets.size());
        CudaBuffer<Color_cuda> d_image(image_width * image_height);
        
        std::vector<Color_cuda> cuda_image(image_width * image_height);;
        
        // Copy data to device
        d_models.copy_from_host(all_triangles.data(), all_triangles.size());
        d_triangle_counts.copy_from_host(triangle_counts.data(), triangle_counts.size());
        d_triangle_offsets.copy_from_host(triangle_offsets.data(), triangle_offsets.size());
        
        // Launch kernel
        launch_render_kernel(
            d_image.get(),
            image_width,
            image_height,
            camera.camera_center,
            camera.pixel100_loc,
            camera.pixel_delta_u,
            camera.pixel_delta_v,
            d_models.get(),
            d_triangle_counts.get(),
            d_triangle_offsets.get(),
            s->objects.size(),
            luminosity
        );
        
        // Check for kernel launch errors
        cuda_utils::check_last_error("kernel launch");
        
        // Synchronize device
        cuda_utils::synchronize_device();
        
        // Copy result back
        d_image.copy_to_host(cuda_image.data(), cuda_image.size());
        
        // Convert CUDA colors to regular colors
        for (int i = 0; i < image_width * image_height; ++i) {
            image[i] = cuda_to_color(cuda_image[i]);
        }
        
        // Memory is automatically freed when CudaBuffer objects go out of scope
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA operation failed: " << e.what() << std::endl;
        return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
    } catch (...) {
        std::cerr << "Unknown error during CUDA rendering" << std::endl;
        return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
    }
    
    // Write both PPM and JPG files
    {
        // Determine base filename (without extension)
        std::string base_path(output_path);
        size_t dot_pos = base_path.find_last_of(".");
        if (dot_pos != std::string::npos) {
            base_path = base_path.substr(0, dot_pos);
        }
        
        std::string ppm_path = base_path + ".ppm";
        std::string jpg_path = base_path + ".jpg";
        
        // Write PPM file
        if (!write_ppm(ppm_path, image, image_width, image_height)) {
            std::cerr << "Failed to write PPM image to: " << ppm_path << std::endl;
            return static_cast<int>(RendererError::FILE_WRITE_FAILED);
        }
        
        // Verify PPM file was written
        std::ifstream verify_ppm(ppm_path);
        if (!verify_ppm.good()) {
            std::cerr << "Failed to verify PPM output file: " << ppm_path << std::endl;
            return static_cast<int>(RendererError::FILE_WRITE_FAILED);
        }
        verify_ppm.close();
        
        // Write JPG file
        if (!write_jpg(jpg_path, image, image_width, image_height, 90)) {
            std::cerr << "Failed to write JPG image to: " << jpg_path << std::endl;
            return static_cast<int>(RendererError::FILE_WRITE_FAILED);
        }
        
        // Verify JPG file was written
        std::ifstream verify_jpg(jpg_path);
        if (!verify_jpg.good()) {
            std::cerr << "Failed to verify JPG output file: " << jpg_path << std::endl;
            return static_cast<int>(RendererError::FILE_WRITE_FAILED);
        }
        verify_jpg.close();
    }
    
    return static_cast<int>(RendererError::SUCCESS);
}

// Light management functions
int add_light_to_scene(SceneHandle scene, Light_API* light) {
    if (!scene || !light) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    vec3 pos(light->position.x, light->position.y, light->position.z);
    color col(light->color.x, light->color.y, light->color.z);
    Light l(pos, col, light->luminosity);
    s->lights.push_back(l);
    return static_cast<int>(RendererError::SUCCESS);
}

int remove_light_from_scene(SceneHandle scene, int index) {
    if (!scene) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->lights.size())) {
        return static_cast<int>(RendererError::INVALID_INDEX);
    }
    
    s->lights.erase(s->lights.begin() + index);
    return static_cast<int>(RendererError::SUCCESS);
}

int update_light(SceneHandle scene, int index, Light_API* light) {
    if (!scene || !light) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->lights.size())) {
        return static_cast<int>(RendererError::INVALID_INDEX);
    }
    
    vec3 pos(light->position.x, light->position.y, light->position.z);
    color col(light->color.x, light->color.y, light->color.z);
    s->lights[index] = Light(pos, col, light->luminosity);
    return static_cast<int>(RendererError::SUCCESS);
}

int get_scene_light_count(SceneHandle scene) {
    if (!scene) {
        return 0;
    }
    Scene* s = static_cast<Scene*>(scene);
    return static_cast<int>(s->lights.size());
}

void free_scene(SceneHandle scene) {
    if (scene) {
        delete static_cast<Scene*>(scene);
    }
}

} // extern "C"
