# Ray Tracer Project
Ray Tracer for CS 147 GPU, Final Project. The purpose of the project is to compare the runtimes of ray tracing on CPU vs. GPU. The CPU code is written in C++, and the GPU code is written in NVIDIA CUDA. The ray tracer generates numerous spheres of varying materials and color based on a seed value. Some of the header files are based off of my previous ray tracer for CS 130 Computer Graphics. 

## Hardware Specifications:
### Device and CPU Specifications:
MacBook Pro (15-inch, Mid 2015)<br />
2.8 GHz Intel Core i7-4980HQ CPU<br />
MacOS Mojave - Version 10.14.4<br />
16 GB 1600 MHz DDR3<br />

### GPU Specifications:
University of California, Riverside <br />
Engineering Server Bender<br />

NVIDIA Tesla M60 GPU x 4<br />
NVIDIA-SMI 410.79<br />
Driver Version: 410.79<br />
CUDA Version: 10.0<br />


## Program Statistics:

Precision | CPU Runtime (sec.) | GPU Runtime (sec.) 
------------ | ------------ | ------------
100 | 632.25 | 179.31
50 | 323.98 | 90.44
40 | 251.71 | 77.21
30 | 189.56 | 59.54
20 | 126.60 | 42.29
10 | 64.33 | 24.62


## Sample Images:

#### CUDA Rendered @ 10 Precision (1440 x 900 px)
![alt text](https://raw.githubusercontent.com/Alien002/Ray_Tracer_Project/master/Assets/CUDA_10_Image.png "CUDA_10_Image.png")<br />
*Rendered image is grainy and not high quality.*
<br />
<br />
#### CUDA Rendered @ 100 Precision (1440 x 900 px)
![alt text](https://raw.githubusercontent.com/Alien002/Ray_Tracer_Project/master/Assets/CUDA_100_Image.png "CUDA_100_Image.png")<br />
*Rendered image is not grainy and much higher quality.*
