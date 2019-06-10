#ifndef __VEC3_H__
#define __VEC3_H__

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3  {

    
public:
    vec3() {}
    vec3(float a, float b, float c) {v[0] = a; v[1] = b; v[2] = c;}
    inline float x() const { return v[0]; }
    inline float y() const { return v[1]; }
    inline float z() const { return v[2]; }
    inline float r() const { return v[0]; }
    inline float g() const { return v[1]; }
    inline float b() const { return v[2]; }
    
    inline const vec3& operator+() const { return *this; }
    inline vec3 operator-() const { return vec3(-v[0], -v[1], -v[2]); }
    inline float operator[](int i) const { return v[i]; }
    inline float& operator[](int i) { return v[i]; };
    
    inline vec3& operator+=(const vec3 &ve2);
    inline vec3& operator-=(const vec3 &ve2);
    inline vec3& operator*=(const vec3 &ve2);
    inline vec3& operator/=(const vec3 &ve2);
    inline vec3& operator*=(const float t);
    inline vec3& operator/=(const float t);
    
    inline float length() const { return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
    inline float squared_length() const { return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]; }
    inline void make_unit_vector();
    
    
    float v[3];
};



inline std::istream& operator >> (std::istream& infs, vec3 & u)
{
    for(int i = 0; i < 3; i++)
    {
        infs >> u.v[i];
    }
    return infs;
}

inline std::ostream& operator << (std::ostream& outfs, const vec3 & u)
{
    for(int i = 0; i < 3; i++)
    {
        if(i) outfs <<" ";
        outfs << u.v[i];
    }
    return outfs;
}

inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= k; v[1] *= k; v[2] *= k;
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    vec3 r;
    for(int i = 0; i < 3; i++){
        r.v[i] = v1.v[i] + v2.v[i];
    }
    return r;
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    vec3 r;
    for(int i = 0; i < 3; i++){
        r.v[i] = v1.v[i] - v2.v[i];
    }
    return r;
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    vec3 r;
    for(int i = 0; i < 3; i++){
        r.v[i] = v1.v[i] * v2.v[i];
    }
    return r;
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    vec3 r;
    for(int i = 0; i < 3; i++){
        r.v[i] = v1.v[i] / v2.v[i];
    }
    return r;
}

inline vec3 operator*(float t, const vec3 &v1) {
    return vec3(t*v1.v[0], t*v1.v[1], t*v1.v[2]);
}

inline vec3 operator/(vec3 v1, float t) {
    return vec3(v1.v[0]/t, v1.v[1]/t, v1.v[2]/t);
}

inline vec3 operator*(const vec3 &v1, float t) {
    return vec3(t*v1.v[0], t*v1.v[1], t*v1.v[2]);
}

inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.v[0] *v2.v[0] + v1.v[1] *v2.v[1]  + v1.v[2] *v2.v[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3( (v1.v[1]*v2.v[2] - v1.v[2]*v2.v[1]),
                (-(v1.v[0]*v2.v[2] - v1.v[2]*v2.v[0])),
                (v1.v[0]*v2.v[1] - v1.v[1]*v2.v[0]));
}

inline vec3& vec3::operator += (const vec3& x)
{for(int i = 0; i < 3; i++) v[i] += x.v[i]; return *this;}

inline vec3& vec3::operator -= (const vec3& x)
{for(int i = 0; i < 3; i++) v[i] -= x.v[i]; return *this;}

inline vec3& vec3::operator *= (const vec3& x)
{for(int i = 0; i < 3; i++) v[i] *= x.v[i]; return *this;}

inline vec3& vec3::operator /= (const vec3& x)
{for(int i = 0; i < 3; i++) v[i] /= x.v[i]; return *this;}

inline vec3& vec3::operator *= (const float y)
{for(int i = 0; i < 3; i++) v[i] *= y; return *this;}

inline vec3& vec3::operator /= (const float y)
{for(int i = 0; i < 3; i++) v[i] /= y; return *this;}

inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif
