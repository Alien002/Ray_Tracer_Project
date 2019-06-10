#ifndef __RAY_H__
#define __RAY_H__

#include "vec3.h"

class ray
{
public:
    __device__ vec3 endpoint() const { return vec_endpoint;}       // endpoint of the ray where t=0
    
    __device__ vec3 direction() const { return vec_direction;}     // direction the ray sweeps out - unit vector

    __device__ ray()
        :vec_endpoint(0,0,0),vec_direction(0,0,1)
    {}

    __device__ ray(const vec3& endpoint_input,const vec3& direction_input)
        :vec_endpoint(endpoint_input), vec_direction(direction_input)
    {}

    __device__ vec3 point_at(float t) const
    {
        return vec_endpoint+(vec_direction*t);
    }
    
private:                                                //privatized endpoint and directions
    vec3 vec_endpoint;
    vec3 vec_direction;
};
#endif
