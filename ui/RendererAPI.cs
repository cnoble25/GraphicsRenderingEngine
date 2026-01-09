using System;
using System.Runtime.InteropServices;

namespace GraphicsRendererUI
{
    // C-compatible structures matching renderer_api.h
    [StructLayout(LayoutKind.Sequential)]
    public struct Vec3_API
    {
        public double x;
        public double y;
        public double z;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Rotation_API
    {
        public double roll;
        public double pitch;
        public double yaw;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Transform_API
    {
        public Vec3_API position;
        public Rotation_API rotation;
        public Vec3_API scale;
    }

    public enum ObjectType_API
    {
        OBJECT_TYPE_PYRAMID = 0,
        OBJECT_TYPE_BOX = 1,
        OBJECT_TYPE_OBJ_FILE = 2,
        OBJECT_TYPE_PLANE = 3
    }

    public enum RenderMode_API
    {
        RENDER_MODE_RAY_TRACING = 0,
        RENDER_MODE_RASTERIZATION = 1
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Light_API
    {
        public Vec3_API position;
        public Vec3_API color;  // RGB color (0.0-1.0 range)
        public double luminosity;  // Light intensity
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SceneObject_API
    {
        public ObjectType_API type;
        public Transform_API transform;
        [MarshalAs(UnmanagedType.LPStr)]
        public string? obj_file_path;  // char* in C, null if not used
        public double light_absorption;  // Light absorption scalar (0.0-1.0), 0.0 = perfect mirror, 1.0 = completely matte
    }

    public static class RendererAPI
    {

#if WINDOWS
        private const string DllName = "GraphicsRendererAPI.dll";
#elif LINUX
        private const string DllName = "libGraphicsRendererAPI.so";
#elif OSX
        private const string DllName = "libGraphicsRendererAPI.dylib";
#else
        private const string DllName = "libGraphicsRendererAPI.so"; // Default to Linux
#endif

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr create_scene();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int add_object_to_scene(IntPtr scene, ref SceneObject_API object_api);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int remove_object_from_scene(IntPtr scene, int index);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int update_object_transform(IntPtr scene, int index, ref Transform_API transform);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int update_object_light_absorption(IntPtr scene, int index, double light_absorption);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int get_scene_object_count(IntPtr scene);

        // Light management
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int add_light_to_scene(IntPtr scene, ref Light_API light);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int remove_light_from_scene(IntPtr scene, int index);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int update_light(IntPtr scene, int index, ref Light_API light);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int get_scene_light_count(IntPtr scene);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int render_scene(IntPtr scene, string output_path, int width, int height, double luminosity, RenderMode_API render_mode, int max_bounces, int compression_level);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int render_scene_to_buffer(IntPtr scene, [Out] byte[] buffer, int width, int height, int max_bounces, int focus_x, int focus_y, int compression_level);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void free_scene(IntPtr scene);
    }
}
