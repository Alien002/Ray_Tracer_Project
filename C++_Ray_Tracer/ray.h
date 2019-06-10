#ifndef __RAY_H__
#define __RAY_H__

#include "vec3.h"

class ray
{
public:
    
    
    vec3 endpoint() const { return vec_endpoint;}       // endpoint of the ray where t=0
    
    vec3 direction() const { return vec_direction;} // direction the ray sweeps out - unit vector

    ray()
        :vec_endpoint(0,0,0),vec_direction(0,0,1)
    {}

    ray(const vec3& endpoint_input,const vec3& direction_input)
        :vec_endpoint(endpoint_input), vec_direction(direction_input)
    {}

    vec3 point(float t) const
    {
        return vec_endpoint+(vec_direction*t);
    }
private:
    vec3 vec_endpoint;
    vec3 vec_direction;
};
#endif
