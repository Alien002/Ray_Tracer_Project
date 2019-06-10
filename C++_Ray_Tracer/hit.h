//
//  hit.h
//  C++_Ray_Tracer
//
//  Created by Alic Lien on 6/1/19.
//

#ifndef __HIT_H__
#define __HIT_H__

#include "ray.h"

class material;

struct hit_data{
    float t;
    vec3 point;
    vec3 normal;
    material *mat_ptr;
};

class hitable {
    public:
    virtual bool hit(const ray& r, float t_min, float t_max, hit_data& rec) const = 0;
};


#endif /* hit_h */
