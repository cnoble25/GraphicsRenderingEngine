//
// Created by carso on 12/2/2024.
//

#ifndef TRANSFORM_H
#define TRANSFORM_H
#include "vec3.h"
#include "rotation.h"

class transforms {
public:
    vec3 position;
    rotations rotation;
    vec3 scale;

    transforms(): position(vec3(0,0,0)), rotation(rotations()), scale(vec3(1,1,1)) {
    }

    transforms(vec3 const & pos, rotations const & rot, vec3 const & scl): position(pos), rotation(rot), scale(scl) {
    }






};



#endif // TRANSFORM_H
