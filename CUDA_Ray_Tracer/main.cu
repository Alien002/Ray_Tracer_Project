//
//  main.cpp
//  C++_Ray_Tracer
//
//  Created by Alic Lien on 6/1/19.
//

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include "kernel.cu"

/*
 #include <cfloat>
 #include <curand_kernel.h>
 
 #include "sphere.h"
 #include "camera.h"
 #include "material.h"
 #include "hit_list.h"
 */

using namespace std;

int main(){
    // Store the clock timer @ Funtion Start
    clock_t begin = clock();
    //string filename = "CUDA_50_Image.ppm";
    
    
    // Open a ppm file to store image data
    ofstream outfs ("CUDA_100_Image.ppm");
    if(outfs.is_open()){
        int pixel_x = 1440;      //Pixels on X
        int pixel_y = 900;       //Pixels on Y
        int precision = 100;      //Increase value to achieve higher precision, sample per img
        const unsigned int BLOCK_SIZE_X = 8;
        const unsigned int BLOCK_SIZE_Y = 8;
        int total_pixels = pixel_x * pixel_y;
        size_t buffer_size = total_pixels * sizeof(vec3);
        
        
        cout <<"Rendering a " <<pixel_x <<" x " <<pixel_y <<" image with " <<precision <<" precision for samples per pixel." <<endl;
        cout << "Using " << BLOCK_SIZE_X << " x " << BLOCK_SIZE_Y << " blocks." <<endl;
        
        // Allocate Mem for Frame Buffer
        vec3 *frame_buffer;
        cudaMallocManaged((void **)&frame_buffer, buffer_size);
        
        // Allocate and set random state / seed value
        curandState *d_rand_state;
        curandState *d_rand_state_world;
        
        cudaMalloc((void **)&d_rand_state, total_pixels*sizeof(curandState));
        cudaMalloc((void **)&d_rand_state_world, 1 * sizeof(curandState));
        
        //Set seed value for World Gen
        rand_init<<<1,1>>>(d_rand_state_world);
        
        cudaDeviceSynchronize();
        
        //Generate World and Camera
        hitable **d_list;
        hitable **d_world;
        camera **d_camera;
        
        int num_hitables = 22*22+1+3;
        
        cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *));
        cudaMalloc((void **)&d_world, sizeof(hitable *));
        cudaMalloc((void **)&d_camera, sizeof(camera *));
        
        create_world<<<1,1>>>(d_list, d_world, d_camera, pixel_x, pixel_y, d_rand_state_world);
        
        cudaDeviceSynchronize();
        
        
        //Render frame buffer
        dim3 DimGrid(pixel_x/BLOCK_SIZE_X+1,pixel_y/BLOCK_SIZE_Y+1);
        dim3 DimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
        
        render_init<<<DimGrid, DimBlock>>>(pixel_x, pixel_y, d_rand_state);
        
        cudaDeviceSynchronize();
        
        render<<<DimGrid, DimBlock>>>(frame_buffer, pixel_x, pixel_y,  precision, d_camera, d_world, d_rand_state);
        
        cudaDeviceSynchronize();
        
        //sets X * Y dimentions of generated picture
        outfs << "P3\n" << pixel_x << " " << pixel_y << "\n255\n";
        for (int i = pixel_y - 1; i >= 0; i--) {
            for (int j = 0; j < pixel_x; j++) {
                size_t pixel_index = i * pixel_x + j;
                
                int ir = int(255.99*frame_buffer[pixel_index].r());
                int ig = int(255.99*frame_buffer[pixel_index].g());
                int ib = int(255.99*frame_buffer[pixel_index].b());
                outfs << ir << " " << ig << " " << ib << "\n";
            }
        }
        
        // Free memory ------------------------------------------------------------
        free_world<<<1,1>>>(d_list, d_world, d_camera);
        cudaFree(d_camera);
        cudaFree(d_world);
        cudaFree(d_list);
        cudaFree(d_rand_state);
        cudaFree(frame_buffer);
        
    }
    
    // Close the file
    outfs.close();
    
    cudaDeviceSynchronize();
    
    // Clock time at end of run
    clock_t end = clock();
    
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
    
    cout<<"Total runtime: " <<elapsed_time <<" seconds." <<endl;
    
    // Open file as binary to find size
    ifstream infs("CUDA_100_Image.ppm", ios::binary | ios::ate);
    cout<<"File size: " <<double(infs.tellg()*0.000001) <<" MB" <<endl;
    infs.close();
    
    
    return 0;
}
