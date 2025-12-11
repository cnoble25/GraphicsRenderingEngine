#include "renderer_api.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing negative positions support..." << std::endl;
    
    // Create a scene
    SceneHandle scene = create_scene();
    assert(scene != nullptr);
    
    // Create an object with negative position
    SceneObject_API obj;
    obj.type = OBJECT_TYPE_PYRAMID;
    obj.transform.position.x = -5.0;
    obj.transform.position.y = -3.0;
    obj.transform.position.z = -10.0;
    obj.transform.rotation.roll = 0.0;
    obj.transform.rotation.pitch = 0.0;
    obj.transform.rotation.yaw = 0.0;
    obj.transform.scale.x = 1.0;
    obj.transform.scale.y = 1.0;
    obj.transform.scale.z = 1.0;
    obj.obj_file_path = nullptr;
    
    // Add object to scene
    int result = add_object_to_scene(scene, &obj);
    assert(result != 0);  // Should succeed
    
    std::cout << "✓ Successfully added object with negative position (-5, -3, -10)" << std::endl;
    
    // Test updating to different negative values
    Transform_API transform;
    transform.position.x = -10.5;
    transform.position.y = -20.0;
    transform.position.z = -15.75;
    transform.rotation.roll = 0.0;
    transform.rotation.pitch = 0.0;
    transform.rotation.yaw = 0.0;
    transform.scale.x = 1.0;
    transform.scale.y = 1.0;
    transform.scale.z = 1.0;
    
    result = update_object_transform(scene, 0, &transform);
    assert(result != 0);  // Should succeed
    
    std::cout << "✓ Successfully updated object to negative position (-10.5, -20.0, -15.75)" << std::endl;
    
    // Test with mixed positive and negative
    transform.position.x = 5.0;
    transform.position.y = -5.0;
    transform.position.z = 0.0;
    
    result = update_object_transform(scene, 0, &transform);
    assert(result != 0);  // Should succeed
    
    std::cout << "✓ Successfully updated object to mixed position (5.0, -5.0, 0.0)" << std::endl;
    
    // Test light with negative position
    Light_API light;
    light.position.x = -2.0;
    light.position.y = -1.0;
    light.position.z = -5.0;
    light.color.x = 1.0;
    light.color.y = 1.0;
    light.color.z = 1.0;
    light.luminosity = 5.0;
    
    result = add_light_to_scene(scene, &light);
    assert(result != 0);  // Should succeed
    
    std::cout << "✓ Successfully added light with negative position (-2, -1, -5)" << std::endl;
    
    // Cleanup
    free_scene(scene);
    
    std::cout << "\n✅ All negative position tests passed!" << std::endl;
    std::cout << "Negative positions are fully supported in the rendering engine." << std::endl;
    
    return 0;
}
