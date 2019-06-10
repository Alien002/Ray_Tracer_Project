#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "ray.h"
#include "hit.h"

struct hit_data;

//Schlick's Approximation for specular reflection
float schlick(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    
    r0 = r0*r0;
    
    return r0 + (1-r0)*pow((1 - cosine),5);
}

//Calculates refraction with glass
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dp = dot(uv, n);
    float D = 1.0 - ni_over_nt*ni_over_nt*(1-dp*dp);            //Discriminant
    
    if (D > 0) {
        refracted = ni_over_nt*(uv - n*dp) - n*sqrt(D);
        return true;
    }
    
    return false;
}

//Calculates reflection of glass
vec3 reflect(const vec3& v1, const vec3& n) {
     return v1 - 2*dot(v1,n)*n;
}


vec3 random_sphere() {
    vec3 p;
    
    do {
        p = 2.0*vec3(drand48(),drand48(),drand48()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);
    
    return p;
}


class material  {
    public:
        virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered) const = 0;
};

class matte : public material {
    public:
        vec3 diffuse;
        matte() {}
        matte(const vec3& a) : diffuse(a) {}
    
        virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered) const  {
             vec3 target = rec.point + rec.normal + random_sphere();
             scattered = ray(rec.point, target-rec.point);
             attenuation = diffuse;
             return true;
        }

};

class metal : public material {
    public:
        float fuzz;
        vec3 diffuse;

        metal() {}
        metal(const vec3& a, float f) : diffuse(a) {
            if (f < 1){
                fuzz = f;
            }
            else{
                fuzz = 1;
            }
        }
    
        virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.point, reflected + fuzz*random_sphere());
            attenuation = diffuse;
            
            return (dot(scattered.direction(), rec.normal) > 0);
        }
};

class glass : public material {
    public:
        float ref_idx;
    
        glass() {}
        glass(float ref) : ref_idx(ref) {}
    
        virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered) const  {
            vec3 outward_normal;
            vec3 refracted;
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            attenuation = vec3(1.0, 1.0, 1.0);
            float ni_over_nt;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0){
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                cosine = sqrt(1 - ref_idx*ref_idx*(1-cosine*cosine));
            }
            else{
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
            }
            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
                reflect_prob = schlick(cosine, ref_idx);
            }
            else{
                reflect_prob = 1.0;
            }
            if (drand48() < reflect_prob){
                scattered = ray(rec.point, reflected);
            }
            else{
                scattered = ray(rec.point, refracted);
            }
            return true;
        }
};




#endif




