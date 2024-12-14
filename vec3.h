//
// Created by carso on 10/18/2024.
//

#ifndef vec_H
#define vec_H

#include <cmath>
#include <iostream>
#include <vector>


class vec3 {
public:
    double e[3];
    vec3() : e{0,0,0} {}
    vec3(const double e0, const double e1, const double e2) : e{e0, e1, e2} {}

    [[nodiscard]] double x() const {return e[0]; }
    [[nodiscard]] double y() const { return e[1]; }
    [[nodiscard]] double z() const {return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    bool operator==(const vec3& v) const {
        if(v.e[0] == e[0] && v.e[1] == e[1] && v.e[2] == e[2]) {
            return true;
        } else {
            return false;
        }

    }

    vec3& operator*=(double t) {
        e[0]*=t;
        e[1]*=t;
        e[2]*=t;
        return *this;
    }

    vec3 operator/=(double t) {
        return *this *= 1/t;
    }

    [[nodiscard]] double magnitude() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

};

using point3 = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1]-v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3& u, vec3& v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);
}

inline vec3 operator*(vec3& u, vec3& v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);
}

inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);
}

inline vec3 operator*(double const t, const vec3& v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator/(const vec3& v, double const t) {
    return (1/t) * v;
}


inline double dot(const vec3& u, const vec3& v) {
    return  u.e[0] * v.e[0]
            + u.e[1] * v.e[1]
            + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2]- u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0]*v.e[1] - u.e[1]*v.e[0]);
}

inline vec3 unit_vector(const vec3& v) {
    return v/v.magnitude();
}


#endif //vec_H
