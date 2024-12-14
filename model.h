//
// Created by carso on 12/2/2024.
//

#ifndef model_H
#define MODEL_H

#include <vector>
#include "vertex.h"
#include "ray.h"
#include "color.h"
#include "vec3.h"
#include "transform.h"
#include "rotation.h"

class model {
public:
    std::vector <vertex> vertices;
    std::vector <vertex> transformed_vertices;
    transforms transform;

    model(): vertices({}), transform(transforms()), transformed_vertices({}) {
    }
    model(std::vector <vertex> const &vertexes, transforms const &t): vertices(vertexes), transform(t) {
        for (vertex i : vertices) {
            vertex p = vertex( (transform.scale * transform.rotation.rotate(i.first)) + transform.position, (transform.scale * transform.rotation.rotate(i.second)) + transform.position,   (transform.scale * transform.rotation.rotate(i.third)) + transform.position, color(0,0,0));
            transformed_vertices.push_back(p);
        }
    }


    [[nodiscard]] double intersect(const ray& r) const {
        double min = -1;
        for (vertex i : transformed_vertices) {
            double t = i.ray_intersection(r);
            if (t > 0) {
                if (min == -1) {
                    min = t;
                }
                if (t < min) {
                    min = t;
                }
            }

        }
        if(min > 0) {
            return min;
        }
        return -1;
    }





};

inline model box() {
    vec3 p1 = vec3(1,1,1);
    vec3 p2 = vec3(-1,1,1);
    vec3 p3 = vec3(-1,-1,1);
    vec3 p4 = vec3(1,-1,1);
    vec3 p5 = vec3(-1,-1,-1);
    vec3 p6 = vec3(1,-1,-1);
    vec3 p7 = vec3(-1,1,-1);
    vec3 p8 = vec3(1,1,-1);

    vertex v1 = vertex(p1, p4, p8, color(0,0,0));
    vertex v2 = vertex(p8, p6, p4, color(0,0,0));
    vertex v3 = vertex(p1, p7, p8, color(0,0,0));
    vertex v4 = vertex(p1, p2, p7, color(0,0,0));
    vertex v5 = vertex(p2, p3, p7, color(0,0,0));
    vertex v6 = vertex(p3, p5, p7, color(0,0,0));
    vertex v7 = vertex(p1, p3, p4, color(0,0,0));
    vertex v8 = vertex(p5, p6, p7, color(0,0,0));
    vertex v9 = vertex(p1, p2, p3, color(0,0,0));
    vertex v10 = vertex(p6, p7, p8, color(0,0,0));
    vertex v11 = vertex(p3, p4, p6, color(0,0,0));
    vertex v12 = vertex(p3, p5, p6, color(0,0,0));
    model m = model({v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12}, transforms(vec3(0,0,0), rotations(0, 0, 0), vec3(1,1,1)));
    return m;
}

inline model pyamid() {
    vec3 transf = vec3(0,0,10);
    vec3 v1 = vec3(1,0,1);
    vec3 v2 = vec3(-1,0,1);
    vec3 v3 = vec3(-1,0,-1);
    vec3 v4 = vec3(1,0,-1);
    vec3 v5 = vec3(0,2,0);

    vertex t1 = vertex(v1,v2,v3,color());
    vertex t2 = vertex(v1,v4,v3,color());
    vertex t3 = vertex(v1,v2,v5,color());
    vertex t4 = vertex(v1,v4,v5,color());
    vertex t5 = vertex(v2,v3,v5,color());
    vertex t6 = vertex(v3,v4,v5,color());
    model m = model({t1,t2, t3, t4, t5, t6}, transforms(vec3(0,0,10), rotations(0, 0, 0), vec3(2,2,2)));
    std::clog << m.transformed_vertices[5].third << std::endl;
    return m;
}

#endif //MODEL_H
