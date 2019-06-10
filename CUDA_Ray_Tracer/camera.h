#ifndef __CAMERA_H__
#define __CAMERA_H__

#include <algorithm>
#include "ray.h"


//vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) CUDA version of rand
__device__ vec3 random_disk(curandState *local_rand_state) {
    vec3 point;
    do {
        point = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(point,point) >= 1.0f);
    return point;
}

class camera
{
public:
    // Describes camera in space
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;

    __device__ camera();
    // Camera object
    // vfov is top to bottom in degrees
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) {
        lens_radius = aperture / 2.0f;
        float theta = vfov*M_PI/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }
    
    __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
        vec3 rd = lens_radius*random_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }
};
#endif
