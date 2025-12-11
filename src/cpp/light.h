#ifndef LIGHT_H
#define LIGHT_H

#include "vec3.h"
#include "color.h"

struct Light {
    vec3 position;
    color light_color;
    double luminosity;
    
    Light() : position(vec3(0, 0, 0)), light_color(color(1.0, 1.0, 1.0)), luminosity(1.0) {}
    Light(const vec3& pos, const color& col, double lum) 
        : position(pos), light_color(col), luminosity(lum) {}
};

#endif // LIGHT_H
