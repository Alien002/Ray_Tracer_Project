//
//  hit_list.h
//  C++_Ray_Tracer
//
//  Created by Alic Lien on 6/1/19.
//

#ifndef __HIT_LIST_H__
#define __HIT_LIST_H__

#include "hit.h"

class hit_list: public hitable  {
public:
    hit_list() {}
    hit_list(hitable **l, int n) {list = l; list_size = n; }
    virtual bool hit(const ray& r, float tmin, float tmax, hit_data& rec) const;
    hitable **list;
    int list_size;
};

bool hit_list::hit(const ray& r, float t_min, float t_max, hit_data& rec) const {
    hit_data temp_rec;
    bool on_hit = false;
    double closest_hit = t_max;
    
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_hit, temp_rec)) {
            on_hit = true;
            closest_hit = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return on_hit;
}

#endif /* hit_list_h */
