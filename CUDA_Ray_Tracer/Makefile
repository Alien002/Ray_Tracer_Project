NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE         = CUDA_Ray_Tracer
OBJ         = main.o

default: $(EXE)

main.o: main.cu kernel.cu camera.h hit.h hit_list.h material.h ray.h sphere.h vec3.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) *.ppm
