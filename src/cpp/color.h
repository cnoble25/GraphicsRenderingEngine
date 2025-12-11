

#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// Include stb_image_write header (implementation is in jpg_writer.cpp)
#include "stb_image_write.h"

using color = vec3;

inline void write_color(std::ostream& out, const color& pixel_color) {
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

    out << rbyte << ' ' << gbyte << ' ' << bbyte << "\n";
}

// Write JPG image from color array
// Image data is expected in row-major order, with pixels stored right-to-left within each row
bool write_jpg(const std::string& filename, const std::vector<color>& image, int width, int height, int quality);

// Write PPM (P3 format) image from color array
// Image data is expected in row-major order, with pixels stored right-to-left within each row
bool write_ppm(const std::string& filename, const std::vector<color>& image, int width, int height);

#endif //COLOR_H
