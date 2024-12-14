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

     [[nodiscard]] double ray_intersection(const ray& r) const {
        constexpr double epsilon = std::numeric_limits<double>::epsilon();

        const vec3 edge1 = p[1] - p[0];
        const  vec3 edge2 = p[2] - p[0];
        const vec3 ray_cross_e2 = cross(r.direction(), edge2);
        const double det = dot(edge1, ray_cross_e2);

        if (det > -epsilon && det < epsilon) {
            return -1;    // This ray is parallel to this triangle.
        }

        const double inv_det = 1.0 / det;
        const vec3 s = r.origin() - p[0];
        const double u = inv_det * dot(s, ray_cross_e2);
        int k = (int) (u-1.0);

        if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(k) > epsilon)) {
            return -1;
        }

        const vec3 s_cross_e1 = cross(s, edge1);
        const double v = inv_det * dot(r.direction(), s_cross_e1);
        k = (int) (u + v - 1.0);

        if ((v < 0 && abs(v) > epsilon) || (u + v > 1 && abs(k) > epsilon)) {
            return -1;
        }

        // At this stage we can compute t to find out where the intersection point is on the line.
        const double t = inv_det * dot(edge2, s_cross_e1);

        if (t > epsilon) // ray intersection
        {
            double const distance = (r.origin() + t * r.direction()).magnitude();

            return distance;
        }
        else // This means that there is a line intersection but not a ray intersection.
            return -1;
    }

};

#endif //VERTEX_H
