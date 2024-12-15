//
// Created by carso on 11/19/2024.
//



#ifndef VERTEX_H
#define VERTEX_H

#include "ray.h"
#include "vec3.h"
#include "color.h"


class vertex {


public:
    point3 first;
    point3 second;
    point3 third;
    point3 p[3];
    color c;
    vertex(): first(point3(0,0,0)), second(point3(0,0,0)), third(point3(0,0,0)), c(color(0,0,0))
    {
        p[0] = first;
        p[1] = second;
        p[2] = third;
    }

    vertex(const point3& First_Point, const point3& Second_Point, const point3& Third_Point, const color& color):
    first(First_Point), second(Second_Point), third(Third_Point), p{First_Point, Second_Point, Third_Point}, c(color)
    {
    }


    point3 operator[](const int i) const { return p[i]; }


    [[nodiscard]] double ray_intersection(const ray& r)
    const
    {
        // Compute the plane's normal
        vec3 v0v1 = second - first;
        vec3 v0v2 = third - first;
        // No need to normalize
        vec3 N = cross(v0v1, v0v2); // N

        // Step 1: Finding P

        // Check if the ray and plane are parallel
        float NdotRayDirection = dot(N, r.direction());
        if (abs(NdotRayDirection) < std::numeric_limits<double>::epsilon()) // Almost 0
            return false; // They are parallel, so they don't intersect!

        // Compute d parameter using equation 2
        float d = -dot(N, first);

        // Compute t (equation 3)
        double t = -(dot(N, r.origin()) + d) / NdotRayDirection;

        // Check if the triangle is behind the ray
        if (t < 0) return -1; // The triangle is behind

        // Compute the intersection point using equation 1
        vec3 P = r.origin() + t * r.direction();

        vec3 O = r.origin() + r.direction();

        // Step 2: Inside-Outside Test
        vec3 Ne; // Vector perpendicular to triangle's plane

        // Test sidedness of P w.r.t. edge v0v1
        vec3 v0p = P - first;
        Ne = cross(v0v1,v0p);
        if (dot(N,Ne) < 0) return -1; // P is on the right side

        // Test sidedness of P w.r.t. edge v2v1
        vec3 v2v1 = third - second;
        vec3 v1p = P - second;
        Ne = cross(v2v1,v1p);
        if (dot(N,Ne) < 0) return -1; // P is on the right side

        // Test sidedness of P w.r.t. edge v2v0
        vec3 v2v0 = first - third;
        vec3 v2p = P - third;
        Ne = cross(v2v0,v2p);
        if (dot(N,Ne) < 0) return -1; // P is on the right side

        return (P-O).magnitude(); // The ray hits the triangle
    }
};

#endif //VERTEX_H
