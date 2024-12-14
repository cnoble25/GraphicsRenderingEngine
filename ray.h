//
// Created by carso on 10/21/2024.
//

#ifndef ray_H
#define ray_H

#include "vec3.h"


class ray {
public:
    ray():orig(point3(0,0,0)), dir(vec3(0,0,0)){};

    ray(const point3& origin, const vec3& direction): orig(origin), dir(direction) {}

    [[nodiscard]] const point3& origin() const { return orig; }
    [[nodiscard]] const vec3& direction() const { return dir; }

    [[nodiscard]] point3 at(double const t) const {
        return orig + t*dir;
    }


private:
    point3 orig;
    vec3 dir;

};

#endif //RAY_H
