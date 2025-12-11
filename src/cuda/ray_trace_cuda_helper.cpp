#include "ray_trace_cuda.h"
#include "model.h"
#include <vector>

void prepare_models_for_cuda(
    const std::vector<model>& models,
    std::vector<Vertex_cuda>& all_triangles,
    std::vector<int>& triangle_counts,
    std::vector<int>& triangle_offsets
) {
    all_triangles.clear();
    triangle_counts.clear();
    triangle_offsets.clear();
    
    int current_offset = 0;
    
    for (const auto& m : models) {
        // Use getter to ensure transformed vertices are computed (lazy evaluation)
        const auto& transformed = m.get_transformed_vertices();
        const int count = static_cast<int>(transformed.size());
        triangle_counts.push_back(count);
        triangle_offsets.push_back(current_offset);
        
        for (const auto& v : transformed) {
            all_triangles.push_back(vertex_to_cuda(v));
        }
        
        current_offset += count;
    }
}
