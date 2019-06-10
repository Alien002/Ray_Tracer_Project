#ifndef __SPHERE_H__
#define __SPHERE_H__

#include "hit.h"

class sphere : public hitable
{
    public:
        vec3 center;
        float radius;
        material *mat_ptr;
        
        sphere() {}
        
        sphere(const vec3& center_input,float radius_input, material *m)
            :center(center_input),radius(radius_input), mat_ptr(m)
        {}
        
        virtual bool hit(const ray& r, float tmin, float tmax, hit_data& rec) const;
};

bool sphere::hit(const ray& r, float t_min, float t_max, hit_data& rec) const {
    vec3 ray1 = r.endpoint() - center;
    
    //-b +- sqrt(b^2 - 4*a*c) / 2a
    //sqrt(4)/2 = 1
    //-b +- sqrt(b^2 - a*c) / a
    
    float a = dot(r.direction(), r.direction());
    float b = dot(ray1, r.direction());
    float c = dot(ray1, ray1) - radius*radius;
    float D = b*b - a*c;          //discriminant (b^2 - 4ac)
    
    if (D > 0) {
        float temp = (-b - sqrt(D)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = r.point_at(rec.t);
            rec.normal = (rec.point - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(D)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.point = r.point_at(rec.t);
            rec.normal = (rec.point - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
 
    return false;
}

#endif
