//
// Created by carso on 12/9/2024.
//

#ifndef ROTATION_H
#define ROTATION_H
#include "vec3.h"
#include <cmath>
#include <vector>

class rotations {
    public:
        double roll, pitch, yaw;

    [[nodiscard]] double x() const {
        return roll;
    }

    [[nodiscard]] double y() const {
        return pitch;
    }

    [[nodiscard]] double z() const {
        return yaw;
    }

    rotations(): roll(0), pitch(0), yaw(0) {
    }

    rotations(double const rolls, double const pitchs, double const yaws): roll(rolls), pitch(pitchs), yaw(yaws) {
    }



    [[nodiscard]] vec3 rotate(double const x,double const y,double const z) const {
        double const cosyaw = cos(yaw);
        double const sinyaw = sin(yaw);
        double const cospitch = cos(pitch);
        double const sinpitch = sin(pitch);
        double const cosroll = cos(roll);
        double const sinroll = sin(roll);
        double const matrix[3][3] = {
            {cosyaw*cospitch, cosyaw*sinpitch*sinroll - sinyaw*cosroll, cosyaw*sinpitch*cosroll + sinyaw*sinroll},
            {sinyaw*cospitch, sinyaw*sinpitch*sinroll + cosyaw*cosroll, sinyaw*sinpitch*cosroll - cosyaw*sinroll},
            {-sinpitch, cospitch*sinroll, cospitch*cosroll}
        };

        double const x1 = matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z;
        double const y1 = matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z;
        double const z1 = matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z;
        return vec3(x1,y1,z1);


    }
    [[nodiscard]] vec3 rotate(vec3& v) const {
        double const cosyaw = cos(yaw);
        double const sinyaw = sin(yaw);
        double const cospitch = cos(pitch);
        double const sinpitch = sin(pitch);
        double const cosroll = cos(roll);
        double const sinroll = sin(roll);
        double const matrix[3][3] = {
            {cosyaw*cospitch, cosyaw*sinpitch*sinroll - sinyaw*cosroll, cosyaw*sinpitch*cosroll + sinyaw*sinroll},
            {sinyaw*cospitch, sinyaw*sinpitch*sinroll + cosyaw*cosroll, sinyaw*sinpitch*cosroll - cosyaw*sinroll},
            {-sinpitch, cospitch*sinroll, cospitch*cosroll}
        };

        double const x1 = matrix[0][0]*v.x() + matrix[0][1]*v.y() + matrix[0][2]*v.z();
        double const y1 = matrix[1][0]*v.x() + matrix[1][1]*v.y() + matrix[1][2]*v.z();
        double const z1 = matrix[2][0]*v.x() + matrix[2][1]*v.y() + matrix[2][2]*v.z();
        return vec3(x1,y1,z1);


    }
    private:


};


#endif //ROTATION_H
