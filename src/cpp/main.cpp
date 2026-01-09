// Corresponding header (if any)
// System headers
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cstring>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#endif
// Third-party headers
#include <cuda_runtime.h>
#include <SDL2/SDL.h>
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
        
        // Allocate empty lights and light absorptions (main.cpp doesn't use lights)
        std::vector<Light_cuda> empty_lights;
        std::vector<double> empty_light_absorptions(scene_objects.size(), 1.0); // Default to matte
        size_t lights_size = empty_lights.empty() ? 1 : empty_lights.size();
        CudaBuffer<Light_cuda> d_lights(lights_size);
        CudaBuffer<double> d_light_absorptions(empty_light_absorptions.size());
        CudaBuffer<Color_cuda> d_image(image_width * image_height);

        std::vector<Color_cuda> image(image_width * image_height);

        // Copy data to device
        d_models.copy_from_host(all_triangles.data(), all_triangles.size());
        d_triangle_counts.copy_from_host(triangle_counts.data(), triangle_counts.size());
        d_triangle_offsets.copy_from_host(triangle_offsets.data(), triangle_offsets.size());
        
        // Copy empty lights (use dummy light for empty case)
        if (!empty_lights.empty()) {
            d_lights.copy_from_host(empty_lights.data(), empty_lights.size());
        } else {
            Light_cuda dummy_light = {};
            d_lights.copy_from_host(&dummy_light, 1);
        }
        d_light_absorptions.copy_from_host(empty_light_absorptions.data(), empty_light_absorptions.size());

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
            d_lights.get(),
            0, // num_lights = 0
            d_light_absorptions.get(),
            5 // max_bounces
        );

        // Check for kernel launch errors
        cuda_utils::check_last_error("kernel launch");

        // Wait for kernel to finish
        cuda_utils::synchronize_device();

        // Copy result back to host
        d_image.copy_to_host(image.data(), image.size());

        // Convert CUDA colors to regular colors
        std::vector<color> color_image(image_width * image_height);
        for (int i = 0; i < image_width * image_height; ++i) {
            color_image[i] = cuda_to_color(image[i]);
        }
        
        // Use SDL for rendering instead of file output
        // Initialize SDL
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            return 1;
        }
        
        // Create SDL window
        SDL_Window* window = SDL_CreateWindow(
            "Graphics Rendering Engine",
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            image_width,
            image_height,
            SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE
        );
        
        if (!window) {
            std::cerr << "SDL window creation failed: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return 1;
        }
        
        // Create SDL renderer
        SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        if (!renderer) {
            std::cerr << "SDL renderer creation failed: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 1;
        }
        
        // Create SDL texture
        SDL_Texture* texture = SDL_CreateTexture(
            renderer,
            SDL_PIXELFORMAT_RGB24,
            SDL_TEXTUREACCESS_STREAMING,
            image_width,
            image_height
        );
        
        if (!texture) {
            std::cerr << "SDL texture creation failed: " << SDL_GetError() << std::endl;
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 1;
        }
        
        // Convert color array to RGB24 format for SDL
        // CUDA stores pixels right-to-left, so we need to flip horizontally for SDL (left-to-right)
        std::vector<unsigned char> pixel_data(image_width * image_height * 3);
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                // CUDA stores right-to-left, so read from (width - 1 - i) for left-to-right output
                int cuda_idx = j * image_width + (image_width - 1 - i);
                int sdl_idx = j * image_width * 3 + i * 3;
                
                const color& c = color_image[cuda_idx];
                double r = std::max(0.0, std::min(1.0, c.x()));
                double g = std::max(0.0, std::min(1.0, c.y()));
                double b = std::max(0.0, std::min(1.0, c.z()));
                
                pixel_data[sdl_idx + 0] = static_cast<unsigned char>(255.999 * r);
                pixel_data[sdl_idx + 1] = static_cast<unsigned char>(255.999 * g);
                pixel_data[sdl_idx + 2] = static_cast<unsigned char>(255.999 * b);
            }
        }
        
        // Update texture and render
        void* pixels;
        int pitch;
        SDL_LockTexture(texture, nullptr, &pixels, &pitch);
        std::memcpy(pixels, pixel_data.data(), pixel_data.size());
        SDL_UnlockTexture(texture);
        
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        
        std::cout << "Rendering complete! Close the window to exit." << std::endl;
        
        // Event loop
        bool quit = false;
        SDL_Event event;
        while (!quit) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = true;
                }
            }
            SDL_Delay(16); // ~60 FPS
        }
        
        // Cleanup
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

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