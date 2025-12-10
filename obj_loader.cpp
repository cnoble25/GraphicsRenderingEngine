#include "obj_loader.h"
#include "vertex.h"
#include "vec3.h"
#include "color.h"
#include "transform.h"
#include "rotation.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

model load_obj_file(const std::string& file_path) {
    std::vector<vertex> vertices;
    std::vector<vec3> positions;
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << file_path << std::endl;
        return model();
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::string type;
        iss >> type;
        
        if (type == "v") {
            // Vertex position
            double x, y, z;
            if (iss >> x >> y >> z) {
                positions.push_back(vec3(x, y, z));
            }
        } else if (type == "f") {
            // Face - can be in format: f v1 v2 v3 or f v1/vt1 v2/vt2 v3/vt3
            std::vector<int> face_indices;
            std::string vertex_str;
            
            while (iss >> vertex_str) {
                // Handle format like "1", "1/2", or "1/2/3"
                size_t slash_pos = vertex_str.find('/');
                if (slash_pos != std::string::npos) {
                    vertex_str = vertex_str.substr(0, slash_pos);
                }
                
                try {
                    int idx = std::stoi(vertex_str);
                    // OBJ files use 1-based indexing
                    if (idx > 0 && idx <= static_cast<int>(positions.size())) {
                        face_indices.push_back(idx - 1);
                    }
                } catch (...) {
                    // Invalid index, skip
                }
            }
            
            // Create triangles from face (triangulate if needed)
            if (face_indices.size() >= 3) {
                // Create triangle fan
                for (size_t i = 1; i < face_indices.size() - 1; i++) {
                    vec3 v0 = positions[face_indices[0]];
                    vec3 v1 = positions[face_indices[i]];
                    vec3 v2 = positions[face_indices[i + 1]];
                    vertices.push_back(vertex(v0, v1, v2, color(0, 0, 0)));
                }
            }
        }
    }
    
    file.close();
    
    if (vertices.empty()) {
        std::cerr << "No triangles found in OBJ file: " << file_path << std::endl;
        return model();
    }
    
    // Create model with default transform (will be applied later)
    transforms default_transform(vec3(0, 0, 0), rotations(0, 0, 0), vec3(1, 1, 1));
    return model(vertices, default_transform);
}
