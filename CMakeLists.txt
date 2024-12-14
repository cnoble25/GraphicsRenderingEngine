cmake_minimum_required(VERSION 3.28)
project(GraphicsRenderer)

set(CMAKE_CXX_STANDARD 17)

add_executable(GraphicsRenderer main.cpp
        VEC3.h
        color.h
        ray.h
        vertex.h
        ray.h
        model.h
        transform.h
        rotation.h
)
