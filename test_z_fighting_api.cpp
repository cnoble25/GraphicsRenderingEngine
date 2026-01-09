// Programmatic test for z-fighting using renderer API
#include "renderer_api.h"
#include <iostream>
#include <cstring>

int main() {
    std::cout << "=== Z-FIGHTING API TEST ===" << std::endl;
    
    // Create scene
    SceneHandle scene = create_scene();
    if (!scene) {
        std::cerr << "Failed to create scene!" << std::endl;
        return 1;
    }
    
    // Add first box
    SceneObject_API box1;
    box1.type = OBJECT_TYPE_BOX;
    box1.transform.position.x = 0.0;
    box1.transform.position.y = 0.0;
    box1.transform.position.z = 0.0;
    box1.transform.rotation.roll = 0.0;
    box1.transform.rotation.pitch = 0.0;
    box1.transform.rotation.yaw = 0.0;
    box1.transform.scale.x = 1.0;
    box1.transform.scale.y = 1.0;
    box1.transform.scale.z = 1.0;
    box1.obj_file_path = nullptr;
    
    if (!add_object_to_scene(scene, &box1)) {
        std::cerr << "Failed to add box1!" << std::endl;
        return 1;
    }
    std::cout << "Added box 1" << std::endl;
    
    // Add second box at same position (should cause z-fighting)
    SceneObject_API box2;
    box2.type = OBJECT_TYPE_BOX;
    box2.transform.position.x = 0.0;
    box2.transform.position.y = 0.0;
    box2.transform.position.z = 0.0;
    box2.transform.rotation.roll = 0.0;
    box2.transform.rotation.pitch = 0.0;
    box2.transform.rotation.yaw = 0.0;
    box2.transform.scale.x = 1.0;
    box2.transform.scale.y = 1.0;
    box2.transform.scale.z = 1.0;
    box2.obj_file_path = nullptr;
    
    if (!add_object_to_scene(scene, &box2)) {
        std::cerr << "Failed to add box2!" << std::endl;
        return 1;
    }
    std::cout << "Added box 2 (same position - should cause z-fighting)" << std::endl;
    
    // Add pyramid
    SceneObject_API pyramid;
    pyramid.type = OBJECT_TYPE_PYRAMID;
    pyramid.transform.position.x = 0.0;
    pyramid.transform.position.y = 0.0;
    pyramid.transform.position.z = 0.0;
    pyramid.transform.rotation.roll = 0.0;
    pyramid.transform.rotation.pitch = 0.0;
    pyramid.transform.rotation.yaw = 0.0;
    pyramid.transform.scale.x = 1.0;
    pyramid.transform.scale.y = 1.0;
    pyramid.transform.scale.z = 1.0;
    pyramid.obj_file_path = nullptr;
    
    if (!add_object_to_scene(scene, &pyramid)) {
        std::cerr << "Failed to add pyramid!" << std::endl;
        return 1;
    }
    std::cout << "Added pyramid" << std::endl;
    
    // Add light
    Light_API light;
    light.position.x = 0.0;
    light.position.y = 5.0;
    light.position.z = 5.0;
    light.color.x = 1.0;
    light.color.y = 1.0;
    light.color.z = 1.0;
    light.luminosity = 5.0;
    
    if (!add_light_to_scene(scene, &light)) {
        std::cerr << "Failed to add light!" << std::endl;
        return 1;
    }
    std::cout << "Added light" << std::endl;
    
    // Render in ray tracing mode
    std::cout << "Rendering in ray tracing mode..." << std::endl;
    const char* output_path = "renders/z_fighting_api_test.ppm";
    int result = render_scene(scene, output_path, 800, 450, 5.0, RENDER_MODE_RAY_TRACING, 3, 1);
    
    if (result) {
        std::cout << "Render successful! Output: " << output_path << std::endl;
    } else {
        std::cerr << "Render failed!" << std::endl;
        return 1;
    }
    
    // Cleanup
    free_scene(scene);
    
    std::cout << "Test complete!" << std::endl;
    return 0;
}
