// Corresponding header (if any)
// System headers
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#endif
// Third-party headers
#include <cuda_runtime.h>
// Project headers
#include "color.h"
#include "vec3.h"
#include "ray.h"
#include "vertex.h"
#include "model.h"
#include "transform.h"
#include "rotation.h"
#include "ray_trace_cuda.h"
#include "cuda_memory.h"
#include "cuda_utils.h"
#include "camera_setup.h"
#include "constants.h"

int main() {
    try {
        // Initialize CUDA using utility function
        cuda_utils::set_device(0);

        // Define scene objects (models) - you can add more models here
        std::vector<model> scene_objects;
        
        // Add a pyramid to the scene
        scene_objects.push_back(pyramid());
        
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
        constexpr double aspect_ratio = constants::DEFAULT_ASPECT_RATIO;
        const int image_width = constants::DEFAULT_IMAGE_WIDTH;
        int image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        // Camera setup using helper function
        const CameraParams camera = setup_camera(image_width, image_height);

        // Allocate device memory using RAII
        CudaBuffer<Vertex_cuda> d_models(all_triangles.size());
        CudaBuffer<int> d_triangle_counts(triangle_counts.size());
        CudaBuffer<int> d_triangle_offsets(triangle_offsets.size());
        CudaBuffer<Color_cuda> d_image(image_width * image_height);

        std::vector<Color_cuda> image(image_width * image_height);
        
        // Determine project root (go up from build directory if needed)
        std::string output_path = "renders/output.ppm";
        std::string renders_dir = "renders";
        
        // If we're in a build directory, go up to project root
        try {
            if (std::filesystem::exists("CMakeCache.txt")) {
                // We're likely in build directory, try parent directory
                if (std::filesystem::exists("../CMakeLists.txt")) {
                    renders_dir = "../renders";
                    output_path = "../renders/output.ppm";
                }
            }
        } catch (const std::exception&) {
            // If filesystem check fails, use default
        }
        
        // Ensure renders directory exists
        try {
            std::filesystem::create_directories(renders_dir);
        } catch (const std::exception& e) {
#if !defined(_WIN32) && !defined(_WIN64)
            mkdir(renders_dir.c_str(), 0755);
#else
            _mkdir(renders_dir.c_str());
#endif
        }

        // Copy data to device
        d_models.copy_from_host(all_triangles.data(), all_triangles.size());
        d_triangle_counts.copy_from_host(triangle_counts.data(), triangle_counts.size());
        d_triangle_offsets.copy_from_host(triangle_offsets.data(), triangle_offsets.size());

        // Launch CUDA kernel
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
            scene_objects.size(),
            constants::DEFAULT_LUMINOSITY
        );

        // Check for kernel launch errors
        cuda_utils::check_last_error("kernel launch");

        // Wait for kernel to finish
        cuda_utils::synchronize_device();

        // Copy result back to host
        d_image.copy_to_host(image.data(), image.size());

        // Convert CUDA colors to regular colors and write both PPM and JPG images
        std::vector<color> color_image(image_width * image_height);
        for (int i = 0; i < image_width * image_height; ++i) {
            color_image[i] = cuda_to_color(image[i]);
        }
        
        // Determine base filename (without extension)
        std::string base_path = output_path;
        size_t dot_pos = base_path.find_last_of(".");
        if (dot_pos != std::string::npos) {
            base_path = base_path.substr(0, dot_pos);
        }
        
        std::string ppm_path = base_path + ".ppm";
        std::string jpg_path = base_path + ".jpg";
        
        // Write PPM file
        if (!write_ppm(ppm_path, color_image, image_width, image_height)) {
            std::cerr << "Failed to write PPM image to " << ppm_path << std::endl;
            return 1;
        }
        std::cout << "PPM image saved to " << ppm_path << std::endl;
        
        // Write JPG file
        if (!write_jpg(jpg_path, color_image, image_width, image_height, 90)) {
            std::cerr << "Failed to write JPG image to " << jpg_path << std::endl;
            return 1;
        }
        std::cout << "JPG image saved to " << jpg_path << std::endl;

        // Memory is automatically freed when CudaBuffer objects go out of scope
        
        // Reset CUDA device
        cuda_utils::reset_device();

        return 0;
    } catch (const std::runtime_error& e) {
        std::cerr << "CUDA operation failed: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}