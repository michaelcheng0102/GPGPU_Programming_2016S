all:
	nvcc -I../utils -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o -ccbin=/usr/bin/g++-4.9
	nvcc -I../utils -std=c++11 -arch=sm_30 -O2 main.cu counting.o -ccbin=/usr/bin/g++-4.9
run:
	./a.out
    
