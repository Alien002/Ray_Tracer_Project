GCC = g++
GCC_FLAGS = -std=c++11 -g -Wall -O3 -I/usr/include/libpng12
EXE = Ray_Tracer
OBJ = main.o

default: $(EXE)

main.o: main.cpp camera.h hit.h hit_list.h material.h ray.h sphere.h vec3.h
	$(GCC) -o $@ -c $(GCC_FLAGS) main.cpp

$(EXE): $(OBJ)
	$(GCC) -o $(EXE) $(OBJ)

clean:
	rm -f rt *.o *.ppm $(EXE) 
