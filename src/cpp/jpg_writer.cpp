// Implementation file for stb_image_write
// This file defines the implementation of stb_image_write functions
// Define STB_IMAGE_WRITE_IMPLEMENTATION before including color.h
// so that when color.h includes stb_image_write.h, it will include the implementation
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "color.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#if !defined(_WIN32) && !defined(_WIN64)
#include <sys/stat.h>
#include <sys/types.h>
#endif

// Write JPG image from color array
// Image data is expected in row-major order, with pixels stored right-to-left within each row
bool write_jpg(const std::string& filename, const std::vector<color>& image, int width, int height, int quality) {
    std::cerr << "write_jpg: Called with filename=" << filename 
              << ", width=" << width << ", height=" << height 
              << ", quality=" << quality << ", image_size=" << image.size() << std::endl;
    
    // Validate inputs
    if (image.size() != static_cast<size_t>(width * height)) {
        std::cerr << "write_jpg: Image size mismatch. Expected " << (width * height) 
                  << " pixels, got " << image.size() << std::endl;
        return false;
    }
    
    if (width <= 0 || height <= 0) {
        std::cerr << "write_jpg: Invalid dimensions: " << width << "x" << height << std::endl;
        return false;
    }
    
    if (quality < 1 || quality > 100) {
        std::cerr << "write_jpg: Invalid quality: " << quality << " (must be 1-100)" << std::endl;
        return false;
    }
    
    // Ensure output directory exists
    try {
        std::filesystem::path file_path(filename);
        std::filesystem::path dir_path = file_path.parent_path();
        if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
            std::filesystem::create_directories(dir_path);
        }
    } catch (const std::exception& e) {
        // If filesystem fails, try fallback method
        size_t last_slash = filename.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            std::string dir_path = filename.substr(0, last_slash);
#if !defined(_WIN32) && !defined(_WIN64)
            mkdir(dir_path.c_str(), 0755);
#else
            _mkdir(dir_path.c_str());
#endif
        }
    }
    
    // Convert color array to RGB byte array
    // CUDA kernel stores pixels at j*width+i where:
    // - i=0 corresponds to rightmost pixel (original_i=image_width)  
    // - i=width-1 corresponds to leftmost pixel (original_i=1)
    // stbi_write_jpg expects pixels in top-to-bottom, left-to-right order
    // So we need to reverse each row to convert from right-to-left to left-to-right
    std::vector<unsigned char> rgb_data(width * height * 3);
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            // Source index: array stores right-to-left, so reverse to get left-to-right
            // For output position i, we want the pixel at position (width-1-i) in the source
            int src_idx = j * width + (width - 1 - i);
            // Destination index: write in left-to-right order (standard for JPG)
            int dst_idx = (j * width + i) * 3;
            
            // Bounds check
            if (src_idx < 0 || src_idx >= static_cast<int>(image.size())) {
                std::cerr << "write_jpg: Source index out of bounds: " << src_idx << std::endl;
                return false;
            }
            
            const color& pixel_color = image[src_idx];
            
            auto r = pixel_color.x();
            auto g = pixel_color.y();
            auto b = pixel_color.z();
            
            // Clamp values to valid range and handle NaN/Inf
            if (std::isnan(r) || std::isinf(r)) r = 0.0;
            if (std::isnan(g) || std::isinf(g)) g = 0.0;
            if (std::isnan(b) || std::isinf(b)) b = 0.0;
            
            // Clamp to [0, 1] range
            r = std::max(0.0, std::min(1.0, r));
            g = std::max(0.0, std::min(1.0, g));
            b = std::max(0.0, std::min(1.0, b));
            
            rgb_data[dst_idx + 0] = static_cast<unsigned char>(int(255.999 * r));
            rgb_data[dst_idx + 1] = static_cast<unsigned char>(int(255.999 * g));
            rgb_data[dst_idx + 2] = static_cast<unsigned char>(int(255.999 * b));
        }
    }
    
    // Delete existing file if it exists to ensure clean write
    try {
        if (std::filesystem::exists(filename)) {
            std::filesystem::remove(filename);
        }
    } catch (const std::exception& e) {
        // If filesystem removal fails, try C-style removal
        std::remove(filename.c_str());
    }
    
    // Write JPG file
    // stbi_write_jpg returns non-zero on success, 0 on failure
    std::cerr << "write_jpg: Calling stbi_write_jpg..." << std::endl;
    int result = stbi_write_jpg(filename.c_str(), width, height, 3, rgb_data.data(), quality);
    std::cerr << "write_jpg: stbi_write_jpg returned " << result << std::endl;
    if (result == 0) {
        std::cerr << "write_jpg: stbi_write_jpg failed to write file: " << filename << std::endl;
        // Check if file exists (might be a permission issue)
        std::ifstream check_file(filename, std::ios::binary);
        if (check_file.good()) {
            std::cerr << "write_jpg: File exists but stbi_write_jpg returned failure" << std::endl;
            check_file.close();
            // Try to read first few bytes to see what's in the file
            check_file.open(filename, std::ios::binary);
            unsigned char header[3];
            check_file.read(reinterpret_cast<char*>(header), 3);
            check_file.close();
            std::cerr << "write_jpg: File header: " 
                      << std::hex << (int)header[0] << " " << (int)header[1] << " " << (int)header[2] 
                      << std::dec << std::endl;
        } else {
            std::cerr << "write_jpg: File was not created" << std::endl;
        }
        return false;
    }
    
    // Verify file was written successfully and has valid JPG header
    std::ifstream verify(filename, std::ios::binary);
    if (!verify.good()) {
        std::cerr << "write_jpg: Failed to verify written file: " << filename << std::endl;
        return false;
    }
    
    // Check file size
    verify.seekg(0, std::ios::end);
    std::streampos file_size = verify.tellg();
    if (file_size <= 0) {
        std::cerr << "write_jpg: Written file is empty: " << filename << std::endl;
        verify.close();
        return false;
    }
    
    // Verify JPG header (should start with FF D8 FF)
    verify.seekg(0, std::ios::beg);
    unsigned char header[3];
    verify.read(reinterpret_cast<char*>(header), 3);
    verify.close();
    
    if (header[0] != 0xFF || header[1] != 0xD8 || header[2] != 0xFF) {
        std::cerr << "write_jpg: Written file does not have valid JPG header. "
                  << "Expected FF D8 FF, got " 
                  << std::hex << (int)header[0] << " " << (int)header[1] << " " << (int)header[2] 
                  << std::dec << std::endl;
        // Delete the invalid file
        try {
            std::filesystem::remove(filename);
        } catch (const std::exception& e) {
            std::remove(filename.c_str());
        }
        return false;
    }
    
    return true;
}

// Write PPM (P3 format) image from color array
// Image data is expected in row-major order, with pixels stored right-to-left within each row
// This matches the CUDA kernel output format
bool write_ppm(const std::string& filename, const std::vector<color>& image, int width, int height) {
    // Validate inputs
    if (image.size() != static_cast<size_t>(width * height)) {
        std::cerr << "write_ppm: Image size mismatch. Expected " << (width * height) 
                  << " pixels, got " << image.size() << std::endl;
        return false;
    }
    
    if (width <= 0 || height <= 0) {
        std::cerr << "write_ppm: Invalid dimensions: " << width << "x" << height << std::endl;
        return false;
    }
    
    // Ensure output directory exists
    try {
        std::filesystem::path file_path(filename);
        std::filesystem::path dir_path = file_path.parent_path();
        if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
            std::filesystem::create_directories(dir_path);
        }
    } catch (const std::exception& e) {
        // If filesystem fails, try fallback method
        size_t last_slash = filename.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            std::string dir_path = filename.substr(0, last_slash);
#if !defined(_WIN32) && !defined(_WIN64)
            mkdir(dir_path.c_str(), 0755);
#else
            _mkdir(dir_path.c_str());
#endif
        }
    }
    
    // Open file for writing
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "write_ppm: Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write PPM header (P3 format)
    out << "P3\n";
    out << width << " " << height << "\n";
    out << "255\n";
    
    // Write pixel data
    // CUDA kernel stores pixels right-to-left within each row, which matches what PPM decoder expects
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            
            // Bounds check
            if (idx < 0 || idx >= static_cast<int>(image.size())) {
                std::cerr << "write_ppm: Index out of bounds: " << idx << std::endl;
                out.close();
                return false;
            }
            
            const color& pixel_color = image[idx];
            
            auto r = pixel_color.x();
            auto g = pixel_color.y();
            auto b = pixel_color.z();
            
            // Clamp values to valid range and handle NaN/Inf
            if (std::isnan(r) || std::isinf(r)) r = 0.0;
            if (std::isnan(g) || std::isinf(g)) g = 0.0;
            if (std::isnan(b) || std::isinf(b)) b = 0.0;
            
            // Clamp to [0, 1] range
            r = std::max(0.0, std::min(1.0, r));
            g = std::max(0.0, std::min(1.0, g));
            b = std::max(0.0, std::min(1.0, b));
            
            int rbyte = int(255.999 * r);
            int gbyte = int(255.999 * g);
            int bbyte = int(255.999 * b);
            
            // Ensure values are in valid byte range
            rbyte = std::max(0, std::min(255, rbyte));
            gbyte = std::max(0, std::min(255, gbyte));
            bbyte = std::max(0, std::min(255, bbyte));
            
            out << rbyte << " " << gbyte << " " << bbyte;
            
            // Add newline after each pixel for readability (optional, but helps with large files)
            if (i == width - 1) {
                out << "\n";
            } else {
                out << " ";
            }
        }
    }
    
    out.close();
    
    // Verify file was written successfully
    std::ifstream verify(filename);
    if (!verify.good()) {
        std::cerr << "write_ppm: Failed to verify written file: " << filename << std::endl;
        return false;
    }
    
    // Check file size
    verify.seekg(0, std::ios::end);
    std::streampos file_size = verify.tellg();
    if (file_size <= 0) {
        std::cerr << "write_ppm: Written file is empty: " << filename << std::endl;
        verify.close();
        return false;
    }
    verify.close();
    
    return true;
}
