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
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#endif
// Third-party headers
#include <cuda_runtime.h>
#include <SDL2/SDL.h>
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
    std::vector<double> object_light_absorptions;  // Light absorption for each object (0.0-1.0)
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
            
        case OBJECT_TYPE_PLANE: {
            model temp = plane();
            base_vertices = temp.vertices;
            break;
        }
            
        default:
            return static_cast<int>(RendererError::INVALID_OBJECT);
    }
    
    // Construct model directly and push it
    s->objects.emplace_back(base_vertices, t);
    
    // Store light absorption (default to 1.0 if not set, meaning completely matte)
    double light_absorption = (object->light_absorption >= 0.0 && object->light_absorption <= 1.0) ? object->light_absorption : 1.0;
    s->object_light_absorptions.push_back(light_absorption);
    
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
    
    // Also rebuild light absorptions vector
    std::vector<double> new_light_absorptions;
    new_light_absorptions.reserve(s->object_light_absorptions.size() - 1);
    
    for (size_t i = 0; i < s->objects.size(); ++i) {
        if (i != static_cast<size_t>(index)) {
            new_objects.push_back(s->objects[i]);
            // Keep corresponding light absorption
            if (i < s->object_light_absorptions.size()) {
                new_light_absorptions.push_back(s->object_light_absorptions[i]);
            }
        }
    }
    
    s->objects = std::move(new_objects);
    s->object_light_absorptions = std::move(new_light_absorptions);
    
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

int update_object_light_absorption(SceneHandle scene, int index, double light_absorption) {
    if (!scene) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    if (index < 0 || index >= static_cast<int>(s->objects.size())) {
        return static_cast<int>(RendererError::INVALID_INDEX);
    }
    
    // Clamp light absorption to [0, 1]
    light_absorption = (light_absorption < 0.0) ? 0.0 : ((light_absorption > 1.0) ? 1.0 : light_absorption);
    
    // Ensure light absorptions vector is the right size
    if (s->object_light_absorptions.size() != s->objects.size()) {
        s->object_light_absorptions.resize(s->objects.size(), 1.0);
    }
    
    // Update light absorption
    s->object_light_absorptions[index] = light_absorption;
    
    return static_cast<int>(RendererError::SUCCESS);
}

int get_scene_object_count(SceneHandle scene) {
    if (!scene) {
        return 0;
    }
    Scene* s = static_cast<Scene*>(scene);
    return static_cast<int>(s->objects.size());
}

// Helper function to perform CUDA rendering with custom compression and return color array
// Returns true on success, false on failure
static bool perform_cuda_rendering_custom(
    Scene* s,
    int image_width,
    int image_height,
    int max_bounces,
    const PixelCoord_API* pixel_groups,
    const int* group_sizes,
    int num_groups,
    std::vector<color>& image
) {
    try {
        // Initialize CUDA using utility function
        cuda_utils::set_device(0);
        
        // Prepare models for CUDA
        std::vector<Vertex_cuda> all_triangles;
        std::vector<int> triangle_counts;
        std::vector<int> triangle_offsets;
        
        if (!s->objects.empty()) {
            prepare_models_for_cuda(s->objects, all_triangles, triangle_counts, triangle_offsets);
        } else {
            std::cerr << "Rendering empty scene - will render background only" << std::endl;
        }
        
        // Prepare light absorptions for CUDA
        std::vector<double> cuda_light_absorptions;
        if (s->object_light_absorptions.size() == s->objects.size()) {
            cuda_light_absorptions = s->object_light_absorptions;
        } else {
            cuda_light_absorptions.resize(s->objects.size(), 1.0);
        }
        
        // Prepare lights for CUDA
        std::vector<Light_cuda> cuda_lights;
        cuda_lights.reserve(s->lights.size());
        for (const auto& light : s->lights) {
            Light_cuda cuda_light;
            cuda_light.position = vec3_to_cuda(light.position);
            cuda_light.color.r = light.light_color.x();
            cuda_light.color.g = light.light_color.y();
            cuda_light.color.b = light.light_color.z();
            cuda_light.luminosity = light.luminosity;
            cuda_lights.push_back(cuda_light);
        }
        
        // Camera setup
        const CameraParams camera = setup_camera(image_width, image_height);
        
        // Allocate device memory
        size_t models_size = all_triangles.empty() ? 1 : all_triangles.size();
        size_t counts_size = triangle_counts.empty() ? 1 : triangle_counts.size();
        size_t offsets_size = triangle_offsets.empty() ? 1 : triangle_offsets.size();
        
        CudaBuffer<Vertex_cuda> d_models(models_size);
        CudaBuffer<int> d_triangle_counts(counts_size);
        CudaBuffer<int> d_triangle_offsets(offsets_size);
        size_t lights_size = cuda_lights.empty() ? 1 : cuda_lights.size();
        CudaBuffer<Light_cuda> d_lights(lights_size);
        size_t light_absorptions_size = cuda_light_absorptions.empty() ? 1 : cuda_light_absorptions.size();
        CudaBuffer<double> d_light_absorptions(light_absorptions_size);
        CudaBuffer<Color_cuda> d_image(image_width * image_height);
        
        // Prepare pixel groups for CUDA
        // Calculate total pixels and offsets
        int total_pixels = 0;
        std::vector<int> group_offsets(num_groups);
        for (int i = 0; i < num_groups; i++) {
            group_offsets[i] = total_pixels;
            total_pixels += group_sizes[i];
        }
        
        // Convert pixel groups to CUDA format (flatten array)
        std::vector<PixelCoord_cuda> cuda_pixel_coords(total_pixels);
        int flat_idx = 0;
        for (int i = 0; i < num_groups; i++) {
            for (int j = 0; j < group_sizes[i]; j++) {
                // pixel_groups is a flat array, need to calculate index
                int group_start = 0;
                for (int k = 0; k < i; k++) {
                    group_start += group_sizes[k];
                }
                cuda_pixel_coords[flat_idx].x = pixel_groups[group_start + j].x;
                cuda_pixel_coords[flat_idx].y = pixel_groups[group_start + j].y;
                flat_idx++;
            }
        }
        
        CudaBuffer<PixelCoord_cuda> d_pixel_coords(total_pixels);
        CudaBuffer<int> d_group_sizes(num_groups);
        CudaBuffer<int> d_group_offsets(num_groups);
        
        std::vector<Color_cuda> cuda_image(image_width * image_height);
        
        // Copy data to device
        if (!all_triangles.empty()) {
            d_models.copy_from_host(all_triangles.data(), all_triangles.size());
            d_triangle_counts.copy_from_host(triangle_counts.data(), triangle_counts.size());
            d_triangle_offsets.copy_from_host(triangle_offsets.data(), triangle_offsets.size());
        } else {
            Vertex_cuda dummy_vertex = {};
            int dummy_int = 0;
            d_models.copy_from_host(&dummy_vertex, 1);
            d_triangle_counts.copy_from_host(&dummy_int, 1);
            d_triangle_offsets.copy_from_host(&dummy_int, 1);
        }
        
        if (!cuda_lights.empty()) {
            d_lights.copy_from_host(cuda_lights.data(), cuda_lights.size());
        } else {
            Light_cuda dummy_light = {};
            d_lights.copy_from_host(&dummy_light, 1);
        }
        
        if (!cuda_light_absorptions.empty()) {
            d_light_absorptions.copy_from_host(cuda_light_absorptions.data(), cuda_light_absorptions.size());
        } else {
            double dummy_light_absorption = 1.0;
            d_light_absorptions.copy_from_host(&dummy_light_absorption, 1);
        }
        
        // Copy pixel groups to device
        d_pixel_coords.copy_from_host(cuda_pixel_coords.data(), total_pixels);
        d_group_sizes.copy_from_host(group_sizes, num_groups);
        d_group_offsets.copy_from_host(group_offsets.data(), num_groups);
        
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch custom kernel
        int num_models_to_render = all_triangles.empty() ? 0 : static_cast<int>(s->objects.size());
        std::cerr << "Launching custom compression kernel with " << num_models_to_render << " models, " 
                  << cuda_lights.size() << " lights, and " << num_groups << " pixel groups" << std::endl;
        
        launch_render_kernel_custom(
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
            num_models_to_render,
            d_lights.get(),
            static_cast<int>(cuda_lights.size()),
            d_light_absorptions.get(),
            max_bounces,
            d_pixel_coords.get(),
            d_group_sizes.get(),
            d_group_offsets.get(),
            num_groups
        );
        
        // Check for kernel launch errors
        cuda_utils::check_last_error("custom kernel launch");
        
        // Synchronize device
        cuda_utils::synchronize_device();
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double render_time_seconds = duration.count() / 1000.0;
        
        // Output timing information
        std::cerr << "Custom compression render completed in " << std::fixed << std::setprecision(3) 
                  << render_time_seconds << " seconds (" << duration.count() << " ms)" << std::endl;
        
        // Copy result back
        d_image.copy_to_host(cuda_image.data(), cuda_image.size());
        
        // Convert CUDA colors to regular colors
        image.resize(image_width * image_height);
        for (int i = 0; i < image_width * image_height; ++i) {
            image[i] = cuda_to_color(cuda_image[i]);
        }
        
        return true;
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA operation failed: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error during CUDA rendering" << std::endl;
        return false;
    }
}

// Helper function to perform CUDA rendering and return color array
// Returns true on success, false on failure
static bool perform_cuda_rendering(
    Scene* s,
    int image_width,
    int image_height,
    int max_bounces,
    int compression_level,
    std::vector<color>& image
) {
    try {
        // Initialize CUDA using utility function
        cuda_utils::set_device(0);
        
        // Prepare models for CUDA
        std::vector<Vertex_cuda> all_triangles;
        std::vector<int> triangle_counts;
        std::vector<int> triangle_offsets;
        
        if (!s->objects.empty()) {
            prepare_models_for_cuda(s->objects, all_triangles, triangle_counts, triangle_offsets);
        } else {
            std::cerr << "Rendering empty scene - will render background only" << std::endl;
        }
        
        // Prepare light absorptions for CUDA
        std::vector<double> cuda_light_absorptions;
        if (s->object_light_absorptions.size() == s->objects.size()) {
            cuda_light_absorptions = s->object_light_absorptions;
        } else {
            // If sizes don't match (e.g., old scene), default all to 1.0 (completely matte)
            cuda_light_absorptions.resize(s->objects.size(), 1.0);
        }
        
        // Prepare lights for CUDA - ONLY use lights from the scene, no default lights
        std::vector<Light_cuda> cuda_lights;
        cuda_lights.reserve(s->lights.size());
        for (const auto& light : s->lights) {
            Light_cuda cuda_light;
            cuda_light.position = vec3_to_cuda(light.position);
            cuda_light.color.r = light.light_color.x();
            cuda_light.color.g = light.light_color.y();
            cuda_light.color.b = light.light_color.z();
            cuda_light.luminosity = light.luminosity;
            cuda_lights.push_back(cuda_light);
            std::cerr << "Added light at (" << light.position.x() << ", " << light.position.y() << ", " << light.position.z() 
                      << ") with luminosity " << light.luminosity << std::endl;
        }
        
        std::cerr << "Total lights in scene: " << cuda_lights.size() << std::endl;
        
        // Camera setup using helper function
        const CameraParams camera = setup_camera(image_width, image_height);
        
        // Allocate device memory using RAII
        // For empty scenes or scenes with no triangles, allocate minimal buffers (size 1) to avoid CUDA errors
        size_t models_size = all_triangles.empty() ? 1 : all_triangles.size();
        size_t counts_size = triangle_counts.empty() ? 1 : triangle_counts.size();
        size_t offsets_size = triangle_offsets.empty() ? 1 : triangle_offsets.size();
        
        CudaBuffer<Vertex_cuda> d_models(models_size);
        CudaBuffer<int> d_triangle_counts(counts_size);
        CudaBuffer<int> d_triangle_offsets(offsets_size);
        // Allocate lights buffer - use size 1 minimum to avoid CUDA errors, but pass actual count
        size_t lights_size = cuda_lights.empty() ? 1 : cuda_lights.size();
        CudaBuffer<Light_cuda> d_lights(lights_size);
        // Allocate light absorptions buffer
        size_t light_absorptions_size = cuda_light_absorptions.empty() ? 1 : cuda_light_absorptions.size();
        CudaBuffer<double> d_light_absorptions(light_absorptions_size);
        CudaBuffer<Color_cuda> d_image(image_width * image_height);
        
        std::vector<Color_cuda> cuda_image(image_width * image_height);
        
        // Copy data to device
        if (!all_triangles.empty()) {
            d_models.copy_from_host(all_triangles.data(), all_triangles.size());
            d_triangle_counts.copy_from_host(triangle_counts.data(), triangle_counts.size());
            d_triangle_offsets.copy_from_host(triangle_offsets.data(), triangle_offsets.size());
        } else {
            // Initialize dummy buffers for empty scenes (kernel won't access them when num_models=0)
            Vertex_cuda dummy_vertex = {};
            int dummy_int = 0;
            d_models.copy_from_host(&dummy_vertex, 1);
            d_triangle_counts.copy_from_host(&dummy_int, 1);
            d_triangle_offsets.copy_from_host(&dummy_int, 1);
        }
        
        // Copy lights to device - only if there are lights
        if (!cuda_lights.empty()) {
            d_lights.copy_from_host(cuda_lights.data(), cuda_lights.size());
        } else {
            // Initialize dummy light (kernel won't use it when num_lights=0)
            Light_cuda dummy_light = {};
            d_lights.copy_from_host(&dummy_light, 1);
        }
        
        // Copy light absorptions to device
        if (!cuda_light_absorptions.empty()) {
            d_light_absorptions.copy_from_host(cuda_light_absorptions.data(), cuda_light_absorptions.size());
        } else {
            double dummy_light_absorption = 1.0;
            d_light_absorptions.copy_from_host(&dummy_light_absorption, 1);
        }
        
        // Launch kernel (with num_models = 0 for empty scenes, which will render just background)
        // Use triangle check to handle both empty scenes and scenes with objects but no triangles
        int num_models_to_render = all_triangles.empty() ? 0 : static_cast<int>(s->objects.size());
        std::cerr << "Launching render kernel with " << num_models_to_render << " models and " << cuda_lights.size() << " lights, compression_level=" << compression_level << std::endl;
        
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
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
            num_models_to_render,
            d_lights.get(),
            static_cast<int>(cuda_lights.size()),
            d_light_absorptions.get(),
            max_bounces,
            compression_level
        );
        
        // Check for kernel launch errors
        cuda_utils::check_last_error("kernel launch");
        
        // Synchronize device
        cuda_utils::synchronize_device();
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double render_time_seconds = duration.count() / 1000.0;
        
        // Copy result back
        d_image.copy_to_host(cuda_image.data(), cuda_image.size());
        
        // Output timing information
        std::cerr << "Render completed in " << std::fixed << std::setprecision(3) << render_time_seconds 
                  << " seconds (" << duration.count() << " ms)" << std::endl;
        
        // Convert CUDA colors to regular colors
        image.resize(image_width * image_height);
        for (int i = 0; i < image_width * image_height; ++i) {
            image[i] = cuda_to_color(cuda_image[i]);
        }
        
        // Memory is automatically freed when CudaBuffer objects go out of scope
        return true;
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA operation failed: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error during CUDA rendering" << std::endl;
        return false;
    }
}

int render_scene(SceneHandle scene, const char* output_path, int image_width, int image_height, double luminosity, RenderMode_API render_mode, int max_bounces, int compression_level) {
    if (!scene || !output_path) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    
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
    
    std::vector<color> image;
    
    // Perform CUDA rendering using helper function
    if (!perform_cuda_rendering(s, image_width, image_height, max_bounces, compression_level, image)) {
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
    
    std::cerr << "Render completed successfully" << std::endl;
    return static_cast<int>(RendererError::SUCCESS);
}

int render_scene_to_buffer(SceneHandle scene, unsigned char* buffer, int width, int height, int max_bounces, int focus_x, int focus_y, int compression_level) {
    if (!scene || !buffer) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    if (width <= 0 || height <= 0) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    // Clamp focus point to valid range
    focus_x = std::max(0, std::min(width - 1, focus_x));
    focus_y = std::max(0, std::min(height - 1, focus_y));
    
    Scene* s = static_cast<Scene*>(scene);
    
    // Start overall timing
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::vector<color> image;
    
    // Perform CUDA rendering using helper function
    if (!perform_cuda_rendering(s, width, height, max_bounces, compression_level, image)) {
        return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
    }
    
    // End overall timing
    auto overall_end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start);
    double render_time_seconds = overall_duration.count() / 1000.0;
    
    // Auto-save the rendered image with timestamp
    try {
        // Create renders directory if it doesn't exist
        std::filesystem::create_directories("renders");
        
        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "renders/render_" << width << "x" << height << "_comp" << compression_level << "_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".jpg";
        std::string output_path = ss.str();
        
        // Write JPG file
        if (write_jpg(output_path, image, width, height, 90)) {
            std::cerr << "Image auto-saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Warning: Failed to auto-save image to: " << output_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Exception while auto-saving image: " << e.what() << std::endl;
    }
    
    // Output timing information
    std::cerr << "Total render time: " << std::fixed << std::setprecision(3) << render_time_seconds 
              << " seconds (" << overall_duration.count() << " ms)" << std::endl;
    
    // Convert color array to RGBA buffer (Avalonia expects BGRA format)
    // CUDA stores pixels right-to-left, so we need to flip horizontally
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            // Read from right-to-left (CUDA order)
            int cuda_idx = j * width + (width - 1 - i);
            // Write to left-to-right (normal order)
            int buffer_idx = (j * width + i) * 4;
            
            const color& c = image[cuda_idx];
            
            // Clamp and convert to bytes
            double r = std::max(0.0, std::min(1.0, c.x()));
            double g = std::max(0.0, std::min(1.0, c.y()));
            double b = std::max(0.0, std::min(1.0, c.z()));
            
            // Write as BGRA (Avalonia format)
            buffer[buffer_idx + 0] = static_cast<unsigned char>(255.999 * b);  // B
            buffer[buffer_idx + 1] = static_cast<unsigned char>(255.999 * g);  // G
            buffer[buffer_idx + 2] = static_cast<unsigned char>(255.999 * r);  // R
            buffer[buffer_idx + 3] = 255;  // A (opaque)
        }
    }
    
    return static_cast<int>(RendererError::SUCCESS);
}

int render_scene_to_buffer_custom(SceneHandle scene, unsigned char* buffer, int width, int height, int max_bounces, int focus_x, int focus_y, const PixelCoord_API* pixel_groups, const int* group_sizes, int num_groups) {
    if (!scene || !buffer) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    if (width <= 0 || height <= 0) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    if (!pixel_groups || !group_sizes || num_groups <= 0) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    // Clamp focus point to valid range
    focus_x = std::max(0, std::min(width - 1, focus_x));
    focus_y = std::max(0, std::min(height - 1, focus_y));
    
    Scene* s = static_cast<Scene*>(scene);
    
    // Start overall timing
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    std::vector<color> image;
    
    // Perform CUDA rendering with custom compression
    if (!perform_cuda_rendering_custom(s, width, height, max_bounces, pixel_groups, group_sizes, num_groups, image)) {
        return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
    }
    
    // End overall timing
    auto overall_end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start);
    double render_time_seconds = overall_duration.count() / 1000.0;
    
    // Auto-save the rendered image with timestamp
    try {
        std::filesystem::create_directories("renders");
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "renders/render_custom_" << width << "x" << height << "_groups" << num_groups << "_" 
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".jpg";
        std::string output_path = ss.str();
        
        if (write_jpg(output_path, image, width, height, 90)) {
            std::cerr << "Image auto-saved to: " << output_path << std::endl;
        } else {
            std::cerr << "Warning: Failed to auto-save image to: " << output_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Exception while auto-saving image: " << e.what() << std::endl;
    }
    
    // Output timing information
    std::cerr << "Total custom compression render time: " << std::fixed << std::setprecision(3) 
              << render_time_seconds << " seconds (" << overall_duration.count() << " ms)" << std::endl;
    
    // Convert color array to RGBA buffer (Avalonia expects BGRA format)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int cuda_idx = j * width + (width - 1 - i);
            int buffer_idx = (j * width + i) * 4;
            
            const color& c = image[cuda_idx];
            
            double r = std::max(0.0, std::min(1.0, c.x()));
            double g = std::max(0.0, std::min(1.0, c.y()));
            double b = std::max(0.0, std::min(1.0, c.z()));
            
            buffer[buffer_idx + 0] = static_cast<unsigned char>(255.999 * b);
            buffer[buffer_idx + 1] = static_cast<unsigned char>(255.999 * g);
            buffer[buffer_idx + 2] = static_cast<unsigned char>(255.999 * r);
            buffer[buffer_idx + 3] = 255;
        }
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

// SDL rendering state
static SDL_Window* sdl_window = nullptr;
static SDL_Renderer* sdl_renderer = nullptr;
static SDL_Texture* sdl_texture = nullptr;
static int sdl_width = 0;
static int sdl_height = 0;

int render_scene_sdl(SceneHandle scene, int image_width, int image_height, double luminosity, RenderMode_API render_mode, int max_bounces, int compression_level) {
    if (!scene) {
        return static_cast<int>(RendererError::INVALID_SCENE_HANDLE);
    }
    
    Scene* s = static_cast<Scene*>(scene);
    
    // Initialize SDL if not already initialized
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
        return static_cast<int>(RendererError::FILE_WRITE_FAILED);
    }
    
    // Clean up existing window if present
    render_scene_sdl_close();
    
    // Create SDL window
    sdl_window = SDL_CreateWindow(
        "Graphics Rendering Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        image_width,
        image_height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
    );
    
    if (!sdl_window) {
        std::cerr << "SDL window creation failed: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return static_cast<int>(RendererError::FILE_WRITE_FAILED);
    }
    
    // Create SDL renderer
    sdl_renderer = SDL_CreateRenderer(sdl_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!sdl_renderer) {
        std::cerr << "SDL renderer creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(sdl_window);
        sdl_window = nullptr;
        SDL_Quit();
        return static_cast<int>(RendererError::FILE_WRITE_FAILED);
    }
    
    // Create SDL texture for pixel data
    sdl_texture = SDL_CreateTexture(
        sdl_renderer,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STREAMING,
        image_width,
        image_height
    );
    
    if (!sdl_texture) {
        std::cerr << "SDL texture creation failed: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(sdl_renderer);
        SDL_DestroyWindow(sdl_window);
        sdl_renderer = nullptr;
        sdl_window = nullptr;
        SDL_Quit();
        return static_cast<int>(RendererError::FILE_WRITE_FAILED);
    }
    
    sdl_width = image_width;
    sdl_height = image_height;
    
    // Render the scene (same CUDA code as render_scene)
    std::vector<color> image(image_width * image_height);
    
    try {
        // Initialize CUDA using utility function
        cuda_utils::set_device(0);
        
        // Prepare models for CUDA
        std::vector<Vertex_cuda> all_triangles;
        std::vector<int> triangle_counts;
        std::vector<int> triangle_offsets;
        
        if (!s->objects.empty()) {
            prepare_models_for_cuda(s->objects, all_triangles, triangle_counts, triangle_offsets);
        } else {
            std::cerr << "Rendering empty scene - will render background only" << std::endl;
        }
        
        // Prepare light absorptions for CUDA
        std::vector<double> cuda_light_absorptions;
        if (s->object_light_absorptions.size() == s->objects.size()) {
            cuda_light_absorptions = s->object_light_absorptions;
        } else {
            cuda_light_absorptions.resize(s->objects.size(), 1.0);
        }
        
        // Prepare lights for CUDA
        std::vector<Light_cuda> cuda_lights;
        cuda_lights.reserve(s->lights.size());
        for (const auto& light : s->lights) {
            Light_cuda cuda_light;
            cuda_light.position = vec3_to_cuda(light.position);
            cuda_light.color.r = light.light_color.x();
            cuda_light.color.g = light.light_color.y();
            cuda_light.color.b = light.light_color.z();
            cuda_light.luminosity = light.luminosity;
            cuda_lights.push_back(cuda_light);
        }
        
        // Camera setup
        const CameraParams camera = setup_camera(image_width, image_height);
        
        // Allocate device memory
        size_t models_size = all_triangles.empty() ? 1 : all_triangles.size();
        size_t counts_size = triangle_counts.empty() ? 1 : triangle_counts.size();
        size_t offsets_size = triangle_offsets.empty() ? 1 : triangle_offsets.size();
        
        CudaBuffer<Vertex_cuda> d_models(models_size);
        CudaBuffer<int> d_triangle_counts(counts_size);
        CudaBuffer<int> d_triangle_offsets(offsets_size);
        size_t lights_size = cuda_lights.empty() ? 1 : cuda_lights.size();
        CudaBuffer<Light_cuda> d_lights(lights_size);
        size_t light_absorptions_size = cuda_light_absorptions.empty() ? 1 : cuda_light_absorptions.size();
        CudaBuffer<double> d_light_absorptions(light_absorptions_size);
        CudaBuffer<Color_cuda> d_image(image_width * image_height);
        
        std::vector<Color_cuda> cuda_image(image_width * image_height);
        
        // Copy data to device
        if (!all_triangles.empty()) {
            d_models.copy_from_host(all_triangles.data(), all_triangles.size());
            d_triangle_counts.copy_from_host(triangle_counts.data(), triangle_counts.size());
            d_triangle_offsets.copy_from_host(triangle_offsets.data(), triangle_offsets.size());
        } else {
            Vertex_cuda dummy_vertex = {};
            int dummy_int = 0;
            d_models.copy_from_host(&dummy_vertex, 1);
            d_triangle_counts.copy_from_host(&dummy_int, 1);
            d_triangle_offsets.copy_from_host(&dummy_int, 1);
        }
        
        if (!cuda_lights.empty()) {
            d_lights.copy_from_host(cuda_lights.data(), cuda_lights.size());
        } else {
            Light_cuda dummy_light = {};
            d_lights.copy_from_host(&dummy_light, 1);
        }
        
        if (!cuda_light_absorptions.empty()) {
            d_light_absorptions.copy_from_host(cuda_light_absorptions.data(), cuda_light_absorptions.size());
        } else {
            double dummy_light_absorption = 1.0;
            d_light_absorptions.copy_from_host(&dummy_light_absorption, 1);
        }
        
        // Launch kernel
        int num_models_to_render = all_triangles.empty() ? 0 : static_cast<int>(s->objects.size());
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
            num_models_to_render,
            d_lights.get(),
            static_cast<int>(cuda_lights.size()),
            d_light_absorptions.get(),
            max_bounces,
            compression_level
        );
        
        cuda_utils::check_last_error("kernel launch");
        cuda_utils::synchronize_device();
        
        // Copy result back
        d_image.copy_to_host(cuda_image.data(), cuda_image.size());
        
        // Convert CUDA colors to regular colors
        for (int i = 0; i < image_width * image_height; ++i) {
            image[i] = cuda_to_color(cuda_image[i]);
        }
        
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA operation failed: " << e.what() << std::endl;
        render_scene_sdl_close();
        return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
    } catch (...) {
        std::cerr << "Unknown error during CUDA rendering" << std::endl;
        render_scene_sdl_close();
        return static_cast<int>(RendererError::CUDA_MALLOC_FAILED);
    }
    
    // Convert color array to RGB24 format for SDL
    // CUDA stores pixels right-to-left, so we need to flip horizontally for SDL (left-to-right)
    std::vector<unsigned char> pixel_data(image_width * image_height * 3);
    
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            // CUDA stores right-to-left, so read from (width - 1 - i) for left-to-right output
            int cuda_idx = j * image_width + (image_width - 1 - i);
            int sdl_idx = j * image_width * 3 + i * 3;
            
            const color& c = image[cuda_idx];
            
            // Clamp and convert to bytes
            double r = std::max(0.0, std::min(1.0, c.x()));
            double g = std::max(0.0, std::min(1.0, c.y()));
            double b = std::max(0.0, std::min(1.0, c.z()));
            
            pixel_data[sdl_idx + 0] = static_cast<unsigned char>(255.999 * r);
            pixel_data[sdl_idx + 1] = static_cast<unsigned char>(255.999 * g);
            pixel_data[sdl_idx + 2] = static_cast<unsigned char>(255.999 * b);
        }
    }
    
    // Update SDL texture with pixel data
    void* pixels;
    int pitch;
    SDL_LockTexture(sdl_texture, nullptr, &pixels, &pitch);
    std::memcpy(pixels, pixel_data.data(), pixel_data.size());
    SDL_UnlockTexture(sdl_texture);
    
    // Render texture to screen
    SDL_RenderClear(sdl_renderer);
    SDL_RenderCopy(sdl_renderer, sdl_texture, nullptr, nullptr);
    SDL_RenderPresent(sdl_renderer);
    
    std::cerr << "Render completed successfully - SDL window displayed" << std::endl;
    return static_cast<int>(RendererError::SUCCESS);
}

int render_scene_sdl_update(int* should_close) {
    if (!sdl_window) {
        if (should_close) *should_close = 1;
        return 1;
    }
    
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            if (should_close) *should_close = 1;
            return 1;
        }
    }
    
    if (should_close) *should_close = 0;
    return 0;
}

void render_scene_sdl_close() {
    if (sdl_texture) {
        SDL_DestroyTexture(sdl_texture);
        sdl_texture = nullptr;
    }
    if (sdl_renderer) {
        SDL_DestroyRenderer(sdl_renderer);
        sdl_renderer = nullptr;
    }
    if (sdl_window) {
        SDL_DestroyWindow(sdl_window);
        sdl_window = nullptr;
    }
}

} // extern "C"
