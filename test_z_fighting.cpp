// Test program to reproduce z-fighting with multiple objects
#include "color.h"
#include "vec3.h"
#include "ray.h"
#include <vector>
#include <cmath>
#include <iostream>
#include "vertex.h"
#include "model.h"
#include "transform.h"
#include "rotation.h"
#include "rasterization.h"
#include "light.h"
#include <fstream>

int main() {
    std::cout << "=== Z-FIGHTING TEST ===" << std::endl;
    
    // Create multiple objects that could cause z-fighting
    std::vector<model> scene_objects;
    
    // Add a box
    scene_objects.push_back(box());
    
    // Add another box at same position (should cause z-fighting)
    scene_objects.push_back(box());
    
    // Add a pyramid
    scene_objects.push_back(pyramid());
    
    std::cout << "Created " << scene_objects.size() << " objects" << std::endl;
    
    // Create lights
    std::vector<Light> lights;
    lights.push_back(Light(vec3(0, 5, 5), color(1.0, 1.0, 1.0), 5.0));
    
    // Render settings
    int width = 800;
    int height = 450;
    std::vector<color> image(width * height);
    
    std::cout << "Rendering scene..." << std::endl;
    
    // Render using rasterization
    rasterize_scene(scene_objects, image, width, height, lights);
    
    std::cout << "Rendering complete!" << std::endl;
    
    // Save output
    std::ofstream outfile("renders/z_fighting_test.ppm");
    outfile << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        int r = static_cast<int>(255.999 * image[i].x());
        int g = static_cast<int>(255.999 * image[i].y());
        int b = static_cast<int>(255.999 * image[i].z());
        outfile << r << " " << g << " " << b << "\n";
    }
    outfile.close();
    
    std::cout << "Image saved to renders/z_fighting_test.ppm" << std::endl;
    
    return 0;
}
