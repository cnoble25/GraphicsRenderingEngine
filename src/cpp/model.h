//
// Created by carso on 12/2/2024.
//

#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "vertex.h"
#include "ray.h"
#include "color.h"
#include "vec3.h"
#include "transform.h"
#include "rotation.h"

class model {
private:
    mutable bool transformed_dirty_;
    mutable std::vector<vertex> transformed_vertices_;
    
    // Ensure transformed vertices are up-to-date
    void ensure_transformed() const {
        if (transformed_dirty_) {
            transformed_vertices_.clear();
            transformed_vertices_.reserve(vertices.size());
            for (const vertex& v : vertices) {
                const vertex p = vertex(
                    transform.scale * transform.rotation.rotate(v.first) + transform.position,
                    transform.scale * transform.rotation.rotate(v.second) + transform.position,
                    transform.scale * transform.rotation.rotate(v.third) + transform.position,
                    color(0, 0, 0)
                );
                transformed_vertices_.push_back(p);
            }
            transformed_dirty_ = false;
        }
    }
    
public:
    std::vector<vertex> vertices;
    transforms transform;

    model(): vertices({}), transform(transforms()), transformed_dirty_(true) {
    }
    
    model(const std::vector<vertex>& vertexes, const transforms& t): 
        vertices(vertexes), transform(t), transformed_dirty_(true) {
        ensure_transformed();  // Compute transformed vertices immediately
    }
    
    // Update transform without rebuilding entire model
    // This is much more efficient than rebuilding the entire vector
    void update_transform(const transforms& new_transform) {
        transform = new_transform;
        transformed_dirty_ = true;  // Mark as dirty, will recompute on next access
    }

    [[nodiscard]] double intersect(const ray& r) const {
        ensure_transformed();  // Ensure vertices are up-to-date
        double min_t = -1.0;
        for (const vertex& v : transformed_vertices_) {
            const double t = v.ray_intersection(r);
            if (t > 0.0) {
                if (min_t < 0.0 || t < min_t) {
                    min_t = t;
                }
            }
        }
        return min_t;
    }
    
    // Get transformed vertices (lazy evaluation)
    // This ensures vertices are computed before access
    [[nodiscard]] const std::vector<vertex>& get_transformed_vertices() const {
        ensure_transformed();
        return transformed_vertices_;
    }
    
    // Legacy accessor for backward compatibility
    // Note: Direct access to transformed_vertices is deprecated, use get_transformed_vertices() instead
    // This accessor ensures lazy evaluation works correctly
    [[nodiscard]] const std::vector<vertex>& transformed_vertices() const {
        return get_transformed_vertices();
    }





};

inline model box() {
    // Create UNIQUE vertices for each triangle to avoid shared vertex issues
    // Each triangle gets its own set of vertices, ensuring proper normals
    
    // Front face (z=1, facing +z): Two triangles with CCW winding
    vec3 f1_v1 = vec3(1,1,1);   // Front-top-right
    vec3 f1_v2 = vec3(-1,1,1);  // Front-top-left
    vec3 f1_v3 = vec3(-1,-1,1); // Front-bottom-left
    vertex v1 = vertex(f1_v1, f1_v2, f1_v3, color(0,0,0));  // CCW from front
    
    vec3 f2_v1 = vec3(1,1,1);   // Front-top-right
    vec3 f2_v2 = vec3(-1,-1,1); // Front-bottom-left
    vec3 f2_v3 = vec3(1,-1,1);  // Front-bottom-right
    vertex v2 = vertex(f2_v1, f2_v2, f2_v3, color(0,0,0));  // CCW from front
    
    // Back face (z=-1, facing -z): Two triangles with CCW winding
    vec3 b1_v1 = vec3(1,1,-1);   // Back-top-right
    vec3 b1_v2 = vec3(-1,1,-1);  // Back-top-left
    vec3 b1_v3 = vec3(-1,-1,-1); // Back-bottom-left
    vertex v3 = vertex(b1_v1, b1_v2, b1_v3, color(0,0,0));  // CCW from back
    
    vec3 b2_v1 = vec3(1,1,-1);   // Back-top-right
    vec3 b2_v2 = vec3(-1,-1,-1); // Back-bottom-left
    vec3 b2_v3 = vec3(1,-1,-1);  // Back-bottom-right
    vertex v4 = vertex(b2_v1, b2_v2, b2_v3, color(0,0,0));  // CCW from back
    
    // Right face (x=1, facing +x): Two triangles with CCW winding
    vec3 r1_v1 = vec3(1,1,1);   // Front-top-right
    vec3 r1_v2 = vec3(1,-1,1);  // Front-bottom-right
    vec3 r1_v3 = vec3(1,-1,-1); // Back-bottom-right
    vertex v5 = vertex(r1_v1, r1_v2, r1_v3, color(0,0,0));  // CCW from right
    
    vec3 r2_v1 = vec3(1,1,1);   // Front-top-right
    vec3 r2_v2 = vec3(1,-1,-1); // Back-bottom-right
    vec3 r2_v3 = vec3(1,1,-1);  // Back-top-right
    vertex v6 = vertex(r2_v1, r2_v2, r2_v3, color(0,0,0));  // CCW from right
    
    // Left face (x=-1, facing -x): Two triangles with CCW winding
    vec3 l1_v1 = vec3(-1,1,1);   // Front-top-left
    vec3 l1_v2 = vec3(-1,1,-1);  // Back-top-left
    vec3 l1_v3 = vec3(-1,-1,-1); // Back-bottom-left
    vertex v7 = vertex(l1_v1, l1_v2, l1_v3, color(0,0,0));  // CCW from left
    
    vec3 l2_v1 = vec3(-1,1,1);   // Front-top-left
    vec3 l2_v2 = vec3(-1,-1,-1); // Back-bottom-left
    vec3 l2_v3 = vec3(-1,-1,1);  // Front-bottom-left
    vertex v8 = vertex(l2_v1, l2_v2, l2_v3, color(0,0,0));  // CCW from left
    
    // Top face (y=1, facing +y): Two triangles with CCW winding
    vec3 t1_v1 = vec3(1,1,1);   // Front-top-right
    vec3 t1_v2 = vec3(1,1,-1);  // Back-top-right
    vec3 t1_v3 = vec3(-1,1,-1); // Back-top-left
    vertex v9 = vertex(t1_v1, t1_v2, t1_v3, color(0,0,0));  // CCW from top
    
    vec3 t2_v1 = vec3(1,1,1);   // Front-top-right
    vec3 t2_v2 = vec3(-1,1,-1); // Back-top-left
    vec3 t2_v3 = vec3(-1,1,1);  // Front-top-left
    vertex v10 = vertex(t2_v1, t2_v2, t2_v3, color(0,0,0)); // CCW from top
    
    // Bottom face (y=-1, facing -y): Two triangles with CCW winding
    vec3 bot1_v1 = vec3(1,-1,1);   // Front-bottom-right
    vec3 bot1_v2 = vec3(-1,-1,1);  // Front-bottom-left
    vec3 bot1_v3 = vec3(-1,-1,-1); // Back-bottom-left
    vertex v11 = vertex(bot1_v1, bot1_v2, bot1_v3, color(0,0,0)); // CCW from bottom
    
    vec3 bot2_v1 = vec3(1,-1,1);   // Front-bottom-right
    vec3 bot2_v2 = vec3(-1,-1,-1); // Back-bottom-left
    vec3 bot2_v3 = vec3(1,-1,-1);  // Back-bottom-right
    vertex v12 = vertex(bot2_v1, bot2_v2, bot2_v3, color(0,0,0));  // CCW from bottom
    
    model m = model({v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12}, transforms(vec3(0,0,0), rotations(0, 3.14, 0), vec3(1,1,1)));
    return m;
}

inline model pyramid() {
    // Create UNIQUE vertices for each triangle to avoid shared vertex issues
    // Each triangle gets its own set of vertices, ensuring proper normals
    
    // Base triangles (viewed from above, y>0): CCW winding order
    // Base triangle 1
    vec3 base1_v1 = vec3(1,0,1);   // Front-right
    vec3 base1_v2 = vec3(-1,0,1);  // Front-left
    vec3 base1_v3 = vec3(-1,0,-1); // Back-left
    vertex t1 = vertex(base1_v1, base1_v2, base1_v3, color());  // CCW from above
    
    // Base triangle 2
    vec3 base2_v1 = vec3(1,0,1);   // Front-right
    vec3 base2_v2 = vec3(-1,0,-1); // Back-left
    vec3 base2_v3 = vec3(1,0,-1);  // Back-right
    vertex t2 = vertex(base2_v1, base2_v2, base2_v3, color());  // CCW from above
    
    // Side triangles (all CCW when viewed from outside)
    // Front face (facing +z)
    vec3 front_v1 = vec3(1,0,1);   // Front-right
    vec3 front_v2 = vec3(-1,0,1);  // Front-left
    vec3 front_v3 = vec3(0,2,0);   // Top (apex)
    vertex t3 = vertex(front_v1, front_v2, front_v3, color());  // CCW from front
    
    // Right face (facing +x)
    vec3 right_v1 = vec3(1,0,1);   // Front-right
    vec3 right_v2 = vec3(1,0,-1);  // Back-right
    vec3 right_v3 = vec3(0,2,0);   // Top (apex)
    vertex t4 = vertex(right_v1, right_v2, right_v3, color());  // CCW from right
    
    // Back face (facing -z)
    vec3 back_v1 = vec3(-1,0,-1);  // Back-left
    vec3 back_v2 = vec3(1,0,-1);   // Back-right
    vec3 back_v3 = vec3(0,2,0);    // Top (apex)
    vertex t5 = vertex(back_v1, back_v2, back_v3, color());  // CCW from back
    
    // Left face (facing -x)
    vec3 left_v1 = vec3(-1,0,1);   // Front-left
    vec3 left_v2 = vec3(-1,0,-1); // Back-left
    vec3 left_v3 = vec3(0,2,0);    // Top (apex)
    vertex t6 = vertex(left_v1, left_v2, left_v3, color());  // CCW from left
    
    model m = model({t1,t2,t3,t4,t5,t6}, transforms(vec3(0,-2,7), rotations(0, 3.14/4, 0), vec3(1,1,1)));
    return m;
}

inline model plane() {
    // Create a simple plane (quad) as two triangles
    // The plane lies in the XZ plane (y=0) and can be scaled/transformed
    // Normal points in +y direction (facing up)
    
    // Triangle 1: CCW when viewed from +y (above)
    vec3 p1_v1 = vec3(-1, 0, 1);   // Front-left
    vec3 p1_v2 = vec3(-1, 0, -1);  // Back-left
    vec3 p1_v3 = vec3(1, 0, 1);    // Front-right
    vertex v1 = vertex(p1_v1, p1_v2, p1_v3, color(0, 0, 0));  // CCW from above
    
    // Triangle 2: CCW when viewed from +y (above)
    vec3 p2_v1 = vec3(1, 0, 1);    // Front-right
    vec3 p2_v2 = vec3(-1, 0, -1);  // Back-left
    vec3 p2_v3 = vec3(1, 0, -1);   // Back-right
    vertex v2 = vertex(p2_v1, p2_v2, p2_v3, color(0, 0, 0));  // CCW from above
    
    model m = model({v1, v2}, transforms(vec3(0, 0, 0), rotations(0, 0, 0), vec3(1, 1, 1)));
    return m;
}

#endif //MODEL_H
