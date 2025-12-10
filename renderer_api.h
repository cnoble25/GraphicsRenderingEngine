#ifndef RENDERER_API_H
#define RENDERER_API_H

#ifdef _WIN32
    #ifdef GRAPHICS_RENDERER_API_EXPORTS
        #define GRAPHICS_RENDERER_API __declspec(dllexport)
    #else
        #define GRAPHICS_RENDERER_API __declspec(dllimport)
    #endif
#else
    #define GRAPHICS_RENDERER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible structures for .NET interop
typedef struct {
    double x, y, z;
} Vec3_API;

typedef struct {
    double roll, pitch, yaw;
} Rotation_API;

typedef struct {
    Vec3_API position;
    Rotation_API rotation;
    Vec3_API scale;
} Transform_API;

typedef enum {
    OBJECT_TYPE_PYRAMID = 0,
    OBJECT_TYPE_BOX = 1,
    OBJECT_TYPE_OBJ_FILE = 2
} ObjectType_API;

typedef enum {
    RENDER_MODE_RAY_TRACING = 0,
    RENDER_MODE_RASTERIZATION = 1
} RenderMode_API;

typedef struct {
    ObjectType_API type;
    Transform_API transform;
    char* obj_file_path;  // Only used if type == OBJECT_TYPE_OBJ_FILE
} SceneObject_API;

// Scene management
typedef void* SceneHandle;

// Create a new scene
GRAPHICS_RENDERER_API SceneHandle create_scene();

// Add an object to the scene
GRAPHICS_RENDERER_API int add_object_to_scene(SceneHandle scene, SceneObject_API* object);

// Remove an object from the scene by index
GRAPHICS_RENDERER_API int remove_object_from_scene(SceneHandle scene, int index);

// Update an object's transform
GRAPHICS_RENDERER_API int update_object_transform(SceneHandle scene, int index, Transform_API* transform);

// Get number of objects in scene
GRAPHICS_RENDERER_API int get_scene_object_count(SceneHandle scene);

// Render the scene to a PPM file
GRAPHICS_RENDERER_API int render_scene(SceneHandle scene, const char* output_path, int width, int height, double luminosity, RenderMode_API render_mode);

// Free the scene
GRAPHICS_RENDERER_API void free_scene(SceneHandle scene);

#ifdef __cplusplus
}
#endif

#endif // RENDERER_API_H
