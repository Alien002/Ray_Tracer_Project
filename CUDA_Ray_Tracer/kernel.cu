//
//  kernel.cpp
//  CUDA_Ray_Tracer
//
//  Created by Alic Lien on 6/1/19.
//

#include <stdio.h>
#include <iostream>
#include <cfloat>
#include <curand_kernel.h>

#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "hit_list.h"


// Sets color of matte object
__device__ vec3 color(const ray& r, hitable **world, curandState* local_rand_state) {
    hit_data rec;
    ray curr_ray = r;
    vec3 curr_attenuation = vec3(1.0,1.0,1.0);
    
    for(int i = 0; i < 50; i++){
        if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(curr_ray, rec, attenuation, scattered, local_rand_state)) {
                curr_attenuation *= attenuation;
                curr_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(curr_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return curr_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0);
}

//generates random seed for rand state
__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

//calls rand_init to generate seed value
__global__ void render_init(int pix_x, int pix_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    //if thread blocks exceed max image size
    if((i >= pix_x) || (j >= pix_y)){
        return;
    }
    
    int pixel_index = j * pix_x + i;
    
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

#define rand (curand_uniform(&local_rand_state))

// creates the world and fills it with spheres
__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int pix_x, int pix_y, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        
        d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, new matte(vec3(0.5, 0.5, 0.5)));
        
        int a = 1;
        
        for(int i = -11; i < 11; i++) {
            for(int j = -11; j < 11; j++) {
                float choose_mat = rand;
                vec3 center(i + rand, 0.2, j + rand);
                
                if(choose_mat < 0.8f) {
                    d_list[a++] = new sphere(center, 0.2, new matte(vec3(rand*rand, rand*rand, rand*rand)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[a++] = new sphere(center, 0.2, new metal(vec3(0.5f*(1.0f+rand), 0.5f*(1.0f+rand), 0.5f*(1.0f+rand)), 0.5f*rand));
                }
                else {
                    d_list[a++] = new sphere(center, 0.2, new glass(1.5));
                }
            }
        }
        d_list[a++] = new sphere(vec3(0, 1,0),  1.0, new glass(1.5));
        d_list[a++] = new sphere(vec3(-4, 1, 0), 1.0, new matte(vec3(0.4, 0.2, 0.1)));
        d_list[a++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hit_list(d_list, 22*22+1+3);
        
        vec3 lookfrom(10,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length(); //???
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, float(pix_x)/float(pix_y), aperture, dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

//Renders the image
__global__ void render(vec3 *fb, int pix_x, int pix_y, int precision, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if((i >= pix_x) || (j >= pix_y)){
        return;
    }
    
    int pixel_index = j * pix_x + i;
    
    curandState local_rand_state = rand_state[pixel_index];
    
    vec3 col(0,0,0);
    
    for(int k = 0; k < precision; k++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(pix_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(pix_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    
    rand_state[pixel_index] = local_rand_state;
    col /= float(precision);
    for(int n = 0; n < 3; n++){
        col[n] = sqrt(col[n]);
    }
    fb[pixel_index] = col;
}

