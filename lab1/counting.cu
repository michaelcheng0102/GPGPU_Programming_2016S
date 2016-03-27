#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#define K 500
#define N 40000000
#define k 10

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
__device__ int table[N][k];


__global__ void buildTable(const char *text, int *pos, int text_size, int depth)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int level = 1 << depth;
    if(idx >= text_size) {
        return;
    }
	if(depth == 0) {
		if( text[idx]  == '\n') {
			table[idx][0] = 0;
		} else {
			table[idx][0] = 1;
		}
	} else {
		if( idx < text_size/level ) {
			if( table[idx*2][depth-1]==1 && table[idx*2+1][depth-1]==1 ) {
				table[idx][depth] = 1;
			}
		}
	}
}


__global__ void countPos(const char *text, int *pos, int text_size)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= text_size) {
        return ;
    }
	
	if(table[idx][0] == 0) {
		// if the it represents newline
		// return to 
		pos[idx] = 0;
	} else {
		int length = 0;
		int index = idx;
		int add_length = 1;
		int depth = 0;
		while(1) {
			if(index <= 0) {
				break;
			}
			if (index%2 == 0) {
				// Tf the child is the left child, will have to travel to the parents' left node.
				// That is, the length will have to be added by add_length;
				index = (index-1);
				length = length + add_length;
                //printf("%2: len=%d, index=%d, add=%d, depth=%d\n", length, index, add_length, depth);
			}
			if(table[(index-1)/2][depth+1] == 1) {
				add_length = add_length*2;
				depth = depth+1;
				index = (index-1)/2;
                //printf("checktable: len=%d, index=%d, add=%d, depth=%d\n", length, index, add_length, depth);
			} else {
                //printf("else1: len=%d, index=%d, add=%d, depth=%d\n", length, index, add_length, depth);
				break;
			}
		}
		while(depth >= 0 && index >= 0 && add_length > 0) {
			if(table[index][depth] == 1) {
				// the parent is 0 and have to do to left child's left node
				index = index*2 - 1;
				length = length + add_length;
				depth = depth - 1;
				add_length = add_length/2;
                //printf("if1: len=%d, index=%d, add=%d, depth=%d\n", length, index, add_length, depth);
			} else {
				// table[idx][depth] == 0 and have to go to the right child
				index = index*2 + 1;
				depth = depth - 1;
				add_length = add_length/2;
                //printf("if2: len=%d, index=%d, add=%d, depth=%d\n", length, index, add_length, depth);
			}
		}
		pos[idx] = length;
	}
	
}

__global__ void test(const char *text, int *pos, int text_size)
{
	int count=0;
	for(int i=0; i<text_size; i++) {
		if(text[i] == '\n')
			count = 0;
		else
			count = count+1;
		pos[i] = count;
	}
}

__global__ void printTable()
{
	printf("print table\n");
	for(int i=0; i<6; i++) {
		for(int j=0; j<100; j++)
			printf("%d", table[j][i]);
		printf("\n");
	}
}
void CountPosition(const char *text, int *pos, int text_size)
{
	int count = text_size/2;
	printf("text size: %d\n", text_size);
    for(int depth=0; depth<k; depth++) {
	    buildTable<<<40000, 1024>>>(text, pos, text_size, depth);
        cudaDeviceSynchronize();
    }
	countPos<<<40000, 1024>>>(text, pos, text_size);
	//test<<<1,1>>>(text, pos, text_size);
}

struct is_one {
    __device__ __host__ bool operator()(const int &check) {
        return check == 1;
    }
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	printf("check\n");
    int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);
    
    printf("Extract Head\n");
	// TODO
    nhead = thrust::count(thrust::device, pos_d, pos_d+text_size, 1);
    thrust::device_vector<int> seq(text_size);
    for(int i=0; i<text_size; i++) {
        seq[i] = i;
        printf("In seq:%d\n", i);
    }
    thrust::copy_if(seq.begin(), seq.end(), pos_d, head_d, is_one());
    printf("Extract Head\n");


	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
