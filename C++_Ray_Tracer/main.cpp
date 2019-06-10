//
//  main.cpp
//  C++_Ray_Tracer
//
//  Created by Alic Lien on 6/1/19.
//

#include <stdio.h>
#include <fstream>
#include <ctime>
#include <cfloat>


#include "sphere.h"
#include "camera.h"
#include "material.h"
#include "hit_list.h"


using namespace std;

// Sets color of matte object
vec3 color(const ray& r, hitable *world, int depth) {
    hit_data rec;
    
    if (world->hit(r, 0.001, FLT_MAX, rec)) {
        ray scattered;
        vec3 attenuation;
        if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            return attenuation*color(scattered, world, depth + 1);
        }
        else {
            return vec3(0,0,0);
        }
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

// Generates spheres for scene
hitable *random_scene_gen() {
    int n = 500;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0, -1000, 0), 1000, new matte(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    
    for (int a = -11; a < 10; a++) {
        for (int b = -11; b < 10; b++) {
            float rand_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48());
            
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (rand_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, new matte(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (rand_mat < 0.95) { // metal
                    //list[i++] = new sphere(center, 0.2, new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                    
                }
                else {  // glass
                    //list[i++] = new sphere(center, 0.2, new glass(1.5));
                }
            }
        }
    }
    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
    
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new glass(1.5));
    //list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new matte(vec3(0.4, 0.2, 0.1)));
    //list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
    
    return new hit_list(list,i);
}


int main(){
    clock_t begin = clock();
    string filename = "CPP_Image.ppm";
    ofstream outfs (filename);
    if(outfs.is_open()){
        int pixel_x = 1440;      //Pixels on X
        int pixel_y = 900;       //Pixels on Y
        int precision = 10;      //Increase value to achieve higher precision
        
        cout <<"Rendering a " <<pixel_x <<" x " <<pixel_y <<" image with " <<precision <<" precision for samples per pixel." <<endl;
        
        outfs << "P3\n" << pixel_x << " " << pixel_y << "\n255\n";        //sets X * Y dimentions of generated picture
        hitable *list[5];
        list[0] = new sphere(vec3(0,0,-1), 0.5, new matte(vec3(0.1, 0.2, 0.5)));
        list[1] = new sphere(vec3(0,-100.5,-1), 100, new matte(vec3(0.8, 0.8, 0.0)));
        list[2] = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
        list[3] = new sphere(vec3(-1,0,-1), 0.5, new glass(1.5));
        list[4] = new sphere(vec3(-1,0,-1), -0.45, new glass(1.5));
        hitable *world = new hit_list(list,5);
        world = random_scene_gen();
        
        vec3 lookfrom(12,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        
        camera cam(lookfrom, lookat, vec3(0,1,0), 30, float(pixel_x)/float(pixel_y), aperture, dist_to_focus);
        
        for (int i = pixel_y - 1; i >= 0; i--) {
            for (int j = 0; j < pixel_x; j++) {
                vec3 col(0, 0, 0);
                for (int k = 0; k < precision; k++) {
                    float u = float(j + drand48()) / float(pixel_x);
                    float v = float(i + drand48()) / float(pixel_y);
                    ray r = cam.get_ray(u, v);
                    col += color(r, world,0);
                }
                col /= float(precision);
                col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
                int ir = int(255.99*col[0]);
                int ig = int(255.99*col[1]);
                int ib = int(255.99*col[2]);
                outfs << ir << " " << ig << " " << ib << "\n";
            }
        }
        
        
        
    }
    outfs.close();
    
    clock_t end = clock();
    
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    
    cout<<"Total runtime: " <<elapsed_time <<" seconds." <<endl;
    ifstream infs(filename, ios::binary | ios::ate);
    
    cout<<"File size: " <<double(infs.tellg()*0.000001) <<" MB" <<endl;
    infs.close();
    
    return 0;
}
