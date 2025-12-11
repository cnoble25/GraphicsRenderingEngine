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



    [[nodiscard]] vec3 rotate(const double x, const double y, const double z) const {
        const double cosyaw = std::cos(yaw);
        const double sinyaw = std::sin(yaw);
        const double cospitch = std::cos(pitch);
        const double sinpitch = std::sin(pitch);
        const double cosroll = std::cos(roll);
        const double sinroll = std::sin(roll);
        const double matrix[3][3] = {
            {cosyaw*cospitch, cosyaw*sinpitch*sinroll - sinyaw*cosroll, cosyaw*sinpitch*cosroll + sinyaw*sinroll},
            {sinyaw*cospitch, sinyaw*sinpitch*sinroll + cosyaw*cosroll, sinyaw*sinpitch*cosroll - cosyaw*sinroll},
            {-sinpitch, cospitch*sinroll, cospitch*cosroll}
        };

        const double x1 = matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z;
        const double y1 = matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z;
        const double z1 = matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z;
        return vec3(x1, y1, z1);
    }
    [[nodiscard]] vec3 rotate(const vec3& v) const {
        return rotate(v.x(), v.y(), v.z());
    }
    private:


};


#endif //ROTATION_H
