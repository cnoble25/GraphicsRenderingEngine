// PPM to JPG Converter Utility
// Converts a PPM (P3 format) file to JPG format
#include "color.h"
#include "stb_image_write.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

// Read PPM (P3 format) file
bool read_ppm(const std::string& filename, std::vector<color>& image, int& width, int& height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open PPM file: " << filename << std::endl;
        return false;
    }
    
    // Read magic number (should be "P3")
    std::string magic;
    file >> magic;
    if (magic != "P3") {
        std::cerr << "Unsupported PPM format: " << magic << ". Only P3 format is supported." << std::endl;
        file.close();
        return false;
    }
    
    // Skip comments and read dimensions
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        if (iss >> width >> height) {
            break;
        }
    }
    
    if (width <= 0 || height <= 0) {
        std::cerr << "Invalid dimensions: " << width << "x" << height << std::endl;
        file.close();
        return false;
    }
    
    // Read max value (should be 255)
    int max_value;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        if (iss >> max_value) {
            break;
        }
    }
    
    if (max_value != 255) {
        std::cerr << "Unsupported max value: " << max_value << ". Only 255 is supported." << std::endl;
        file.close();
        return false;
    }
    
    // Read pixel data row by row
    // PPM file stores pixels right-to-left within each row (matching CUDA kernel output)
    image.resize(width * height);
    
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int r, g, b;
            if (!(file >> r >> g >> b)) {
                std::cerr << "Failed to read pixel at row " << j << ", col " << i << std::endl;
                file.close();
                return false;
            }
            
            // Clamp values to valid range
            r = std::max(0, std::min(255, r));
            g = std::max(0, std::min(255, g));
            b = std::max(0, std::min(255, b));
            
            // Store pixel at position j*width+i (right-to-left order within row)
            // write_jpg will reverse this to left-to-right for JPG output
            int idx = j * width + i;
            image[idx] = color(
                r / 255.0,
                g / 255.0,
                b / 255.0
            );
        }
    }
    
    file.close();
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <input.ppm> [output.jpg] [quality]" << std::endl;
        std::cerr << "  input.ppm  - Input PPM file (P3 format)" << std::endl;
        std::cerr << "  output.jpg - Output JPG file (default: input filename with .jpg extension)" << std::endl;
        std::cerr << "  quality    - JPG quality 1-100 (default: 90)" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file;
    int quality = 90;
    
    // Determine output filename
    if (argc >= 3) {
        output_file = argv[2];
    } else {
        // Use input filename with .jpg extension
        size_t dot_pos = input_file.find_last_of(".");
        if (dot_pos != std::string::npos) {
            output_file = input_file.substr(0, dot_pos) + ".jpg";
        } else {
            output_file = input_file + ".jpg";
        }
    }
    
    // Parse quality if provided
    if (argc >= 4) {
        quality = std::atoi(argv[3]);
        if (quality < 1 || quality > 100) {
            std::cerr << "Invalid quality: " << quality << ". Must be between 1 and 100." << std::endl;
            return 1;
        }
    }
    
    // Read PPM file
    std::vector<color> image;
    int width, height;
    
    std::cout << "Reading PPM file: " << input_file << std::endl;
    if (!read_ppm(input_file, image, width, height)) {
        std::cerr << "Failed to read PPM file" << std::endl;
        return 1;
    }
    
    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Converting to JPG..." << std::endl;
    
    // Convert to JPG
    if (!write_jpg(output_file, image, width, height, quality)) {
        std::cerr << "Failed to write JPG file" << std::endl;
        return 1;
    }
    
    std::cout << "Successfully converted to: " << output_file << std::endl;
    return 0;
}
