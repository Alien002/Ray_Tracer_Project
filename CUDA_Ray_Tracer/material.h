#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "ray.h"
#include "hit.h"

struct hit_data;

//Schlick's Approximation for specular reflection
__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    
    r0 = r0*r0;
    
    return r0 + (1.0f - r0)*pow((1.0f - cosine), 5.0f);
}

//Calculates refraction with glass
__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dp = dot(uv, n);
    float D = 1.0f - ni_over_nt*ni_over_nt*(1-dp*dp);            //Discriminant
    
    if (D > 0) {
        refracted = ni_over_nt*(uv - n*dp) - n*sqrt(D);
        return true;
    }
    
    return false;
}

//Calculates reflection of glass
__device__ vec3 reflect(const vec3& v1, const vec3& n) {
    return v1 - 2.0f*dot(v1,n)*n;
}

//sphere location random
__device__ vec3 random_sphere(curandState *local_rand_state) {
    vec3 point;
    
    do {
        point = 2.0f* vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state)) - vec3(1,1,1);
    } while (point.squared_length() >= 1.0f);
    
    return point;
}


class material  {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

// matte spheres with slight diffuse and reflection
class matte : public material {
public:
    vec3 diffuse;
    __device__ matte() {}
    __device__ matte(const vec3& a) : diffuse(a) {}
    
    __device__ virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
        vec3 target = rec.point + rec.normal + random_sphere(local_rand_state);
        scattered = ray(rec.point, target-rec.point);
        attenuation = diffuse;
        return true;
    }
    
};

//metal spheres with more reflection
class metal : public material {
public:
    float fuzz;
    vec3 diffuse;
    
    __device__ metal() {}
    __device__ metal(const vec3& a, float f) : diffuse(a) {
        if (f < 1){
            fuzz = f;
        }
        else{
            fuzz = 1;
        }
    }
    
    __device__ virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.point, reflected + fuzz*random_sphere(local_rand_state));
        attenuation = diffuse;
        
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
};

//glass with refraction and unique reflections
class glass : public material {
public:
    float ref_idx;
    
    __device__ glass() {}
    __device__ glass(float ref) : ref_idx(ref) {}
    
    __device__ virtual bool scatter(const ray& r_in, const hit_data& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 refracted;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        attenuation = vec3(1.0, 1.0, 1.0);
        float ni_over_nt;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f){
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else{
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)){
            reflect_prob = schlick(cosine, ref_idx);
        }
        else{
            reflect_prob = 1.0f;
        }
        if (curand_uniform(local_rand_state) < reflect_prob){
            scattered = ray(rec.point, reflected);
        }
        else{
            scattered = ray(rec.point, refracted);
        }
        return true;
    }
};




#endif




