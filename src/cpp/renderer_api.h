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
    OBJECT_TYPE_OBJ_FILE = 2,
    OBJECT_TYPE_PLANE = 3
} ObjectType_API;

// Ray tracing is the only rendering mode
typedef enum {
    RENDER_MODE_RAY_TRACING = 0
} RenderMode_API;

typedef struct {
    ObjectType_API type;
    Transform_API transform;
    char* obj_file_path;  // Only used if type == OBJECT_TYPE_OBJ_FILE
    double light_absorption;  // Light absorption scalar (0.0-1.0), determines how much light is absorbed (0.0 = perfect mirror, 1.0 = completely matte)
} SceneObject_API;

// Light structure
typedef struct {
    Vec3_API position;
    Vec3_API color;  // RGB color (0.0-1.0 range)
    double luminosity;  // Light intensity
} Light_API;

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

// Update an object's light absorption
GRAPHICS_RENDERER_API int update_object_light_absorption(SceneHandle scene, int index, double light_absorption);

// Get number of objects in scene
GRAPHICS_RENDERER_API int get_scene_object_count(SceneHandle scene);

// Light management
GRAPHICS_RENDERER_API int add_light_to_scene(SceneHandle scene, Light_API* light);
GRAPHICS_RENDERER_API int remove_light_from_scene(SceneHandle scene, int index);
GRAPHICS_RENDERER_API int update_light(SceneHandle scene, int index, Light_API* light);
GRAPHICS_RENDERER_API int get_scene_light_count(SceneHandle scene);

// Render the scene to a JPG file
// compression_level: Block size for compression (1 = no compression, 2 = 2x2 blocks, 4 = 4x4 blocks, etc.)
GRAPHICS_RENDERER_API int render_scene(SceneHandle scene, const char* output_path, int width, int height, double luminosity, RenderMode_API render_mode, int max_bounces, int compression_level);

// Pixel coordinate structure for custom compression
typedef struct {
    int x, y;  // Pixel coordinates (0-based)
} PixelCoord_API;

// Render the scene directly to a buffer (RGBA format, 4 bytes per pixel)
// Returns 1 on success, error code on failure
// The buffer must be pre-allocated with size: width * height * 4 bytes
// focus_x and focus_y are pixel coordinates clamped to [0, width-1] and [0, height-1] respectively
// compression_level: Block size for compression (1 = no compression, 2 = 2x2 blocks, 4 = 4x4 blocks, etc.)
GRAPHICS_RENDERER_API int render_scene_to_buffer(SceneHandle scene, unsigned char* buffer, int width, int height, int max_bounces, int focus_x, int focus_y, int compression_level);

// Render with custom compression algorithm
// pixel_groups: Array of pixel groups, where each group is an array of PixelCoord_API
// group_sizes: Array of sizes, one per group (how many pixels in each group)
// num_groups: Number of pixel groups (number of CUDA cores to use)
// Each CUDA core will process one group, calculating ray from average position of pixels in that group
GRAPHICS_RENDERER_API int render_scene_to_buffer_custom(SceneHandle scene, unsigned char* buffer, int width, int height, int max_bounces, int focus_x, int focus_y, const PixelCoord_API* pixel_groups, const int* group_sizes, int num_groups);

// Render the scene to an SDL window (returns 0 on success, non-zero on error)
// The window will stay open until the user closes it or render_scene_sdl_update is called with should_close=true
// compression_level: Block size for compression (1 = no compression, 2 = 2x2 blocks, 4 = 4x4 blocks, etc.)
GRAPHICS_RENDERER_API int render_scene_sdl(SceneHandle scene, int width, int height, double luminosity, RenderMode_API render_mode, int max_bounces, int compression_level);

// Update SDL window (call this in a loop for interactive rendering)
// Returns 0 if window should stay open, non-zero if window was closed
GRAPHICS_RENDERER_API int render_scene_sdl_update(int* should_close);

// Close SDL window and cleanup
GRAPHICS_RENDERER_API void render_scene_sdl_close();

// Free the scene
GRAPHICS_RENDERER_API void free_scene(SceneHandle scene);

#ifdef __cplusplus
}
#endif

#endif // RENDERER_API_H
