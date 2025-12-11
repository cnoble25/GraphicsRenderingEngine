#ifndef CONSTANTS_H
#define CONSTANTS_H

// Mathematical constants
namespace constants {
    // Epsilon for floating-point comparisons
    constexpr double EPSILON = 1e-8;
    
    // Default camera parameters
    constexpr double DEFAULT_FOCAL_LENGTH = 1.0;
    constexpr double DEFAULT_VIEWPORT_HEIGHT = 2.0;
    
    // Default rendering parameters
    constexpr double DEFAULT_LUMINOSITY = 5.0;
    constexpr int DEFAULT_IMAGE_WIDTH = 800;
    constexpr int DEFAULT_IMAGE_HEIGHT = 450;
    constexpr double DEFAULT_ASPECT_RATIO = 16.0 / 9.0;
    
    // Color constants
    constexpr double COLOR_MAX = 1.0;
    constexpr double COLOR_MIN = 0.0;
    constexpr int COLOR_BYTE_MAX = 255;
    
    // Image format constants (JPG uses 8-bit color channels)
    constexpr int IMAGE_MAX_VALUE = 255;
}

#endif // CONSTANTS_H
