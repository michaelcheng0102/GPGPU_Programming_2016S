all:
	nvcc -I../utils -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o
	nvcc -I../utils -std=c++11 -arch=sm_30 -O2 main.cu counting.o
run:
	./a.out
    
